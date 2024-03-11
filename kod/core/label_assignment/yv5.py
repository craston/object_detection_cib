from __future__ import annotations

from typing import Sequence
from typing import NamedTuple

from functools import partial

import einops
import torch
import torchvision as tv

from kod.core.types import FeatureShape
from kod.data.detection import DetectionTarget
from kod.core.anchors.info import AnchorBoxInfo
from kod.core.bbox.boxes import CXCYWHBoundingBox


class AssignmentAnchorInfo(NamedTuple):
    ll: AnchorBoxInfo
    ml: AnchorBoxInfo
    hl: AnchorBoxInfo


class AssignmentTargetIndices(NamedTuple):
    samples: torch.Tensor
    anchors: torch.Tensor
    grid_y: torch.Tensor
    grid_x: torch.Tensor


class AssignmentTargetInfo(NamedTuple):
    indices: AssignmentTargetIndices
    labels: torch.Tensor
    gt_boxes: torch.Tensor
    anchors: torch.Tensor
    target_feature_shape: FeatureShape


class AssignmentResult(NamedTuple):
    ll: AssignmentTargetInfo
    ml: AssignmentTargetInfo
    hl: AssignmentTargetInfo


class Yolov5LabelAssigner(object):
    def __init__(
        self,
        anchor_info: AssignmentAnchorInfo,
        threshold: float = 4.0,
    ):
        self.anchor_info = anchor_info
        self.threshold = threshold

        self.off_bias = 0.5  # bias

        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

        self.off = off.float() * self.off_bias

    def _make_batched_targets(
        self,
        stride: int,
        targets: Sequence[DetectionTarget],
    ) -> torch.Tensor:
        # Arrange all the ground truths bounding boxes in a form
        # of matrix with 6 columns
        #
        # A row of this matrix will be
        # sample_id, label_id, cx, cy, w, h
        #
        # cx, cy, w, h will be rescaled for the target feature shape

        # sample_id, label_id, cx, cy, w, h
        TARGET_SIZE = 1 + 1 + 1 + 1 + 1 + 1

        x_scale_factor = 1 / stride
        y_scale_factor = 1 / stride

        normalized_and_formatted_targets = []
        for idx, t in enumerate(targets):
            target_boxes = t.boxes
            labels = t.labels

            bb_targets = torch.zeros(
                (target_boxes.shape[0], TARGET_SIZE),
                device=t.boxes.device,
            )

            if target_boxes.shape[0]:
                bboxes = tv.ops.box_convert(
                    target_boxes,
                    in_fmt="xyxy",
                    out_fmt="cxcywh",
                )
                # scale the bounding boxes to the target feature shape
                bboxes = CXCYWHBoundingBox.scale_tensor(
                    bboxes,
                    x_scale_factor=x_scale_factor,
                    y_scale_factor=y_scale_factor,
                )

                # and stuff them in the batch
                bb_targets[:, 2:6] = bboxes
                bb_targets[:, 1] = labels
                bb_targets[:, 0] = idx

            normalized_and_formatted_targets.append(bb_targets)

        # Note -
        # because of cat if there is any bb_target with shape (0, whatever)
        # i.e. the one with no bounding box
        # it will be anyways eliminated
        return torch.cat(normalized_and_formatted_targets, 0)

    def _repeat_for_anchors(
        self,
        num_of_anchors: int,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # repeat the target tensor for every anchor
        #
        # e.g. if the target tensor is (5, 6)
        # it will become (3, 5, 6)
        # where 3 => number of anchors
        repeated = einops.repeat(
            targets,
            "number_of_targets target_size -> r number_of_targets target_size",
            r=num_of_anchors,
        )

        # we would now want to insert the anchor index in the formatter target
        # as well
        # i.e. (3, 5, 6) should become (3,5,6+5+1)
        #
        # Note -
        # later the tensor will be reshaped and
        # hence it is a good idea to preserve the anchor index in it

        # To do so we prepare the anchor_indice such that they can be easily
        # concatenated with the repeated tensor above

        anchor_indices = torch.arange(num_of_anchors, device=targets.device)
        anchor_indices = einops.rearrange(anchor_indices, "a -> a 1")
        anchor_indices = einops.repeat(
            anchor_indices,
            "num_of_anchors 1 -> num_of_anchors num_of_targets 1",
            num_of_targets=repeated.shape[1],
        )

        return torch.cat((repeated, anchor_indices), dim=-1)

    def _filter_targets(
        self,
        anchor_boxes: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # ratio of widths and heights between gt and anchors
        # Note - None will inject an extra dimentions to make this
        #        operation happen
        wh_ratio = targets[..., 4:6] / anchor_boxes[:, None, :]

        selected_target_indices = (
            torch.max(wh_ratio, 1.0 / wh_ratio).max(dim=2).values < self.threshold
        )

        # Note -
        # this will change the shape of the target
        return targets[selected_target_indices]

    def _incorporate_neighbouring_cells(
        self,
        input_image_shape: FeatureShape,
        stride: int,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        feature_map = torch.tensor(
            FeatureShape(
                width=input_image_shape.width / stride,
                height=input_image_shape.height / stride,
            ),
            dtype=targets.dtype,
            device=targets.device,
        )

        off = self.off.to(targets.device)

        gxy = targets[:, 2:4]  # grid xy
        gxi = feature_map - gxy  # inverse

        j, k = ((gxy % 1 < self.off_bias) & (gxy > 1)).T
        l, m = ((gxi % 1 < self.off_bias) & (gxi > 1)).T
        j = torch.stack((torch.ones_like(j), j, k, l, m))

        targets = targets.repeat((5, 1, 1))[j]
        offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

        return targets, offsets

    def _assign_per_layer(
        self,
        input_image_shape: FeatureShape,
        targets: torch.Tensor,
        anchor_info: AnchorBoxInfo,
    ) -> AssignmentTargetInfo:
        stride = anchor_info.stride

        batched_targets = self._make_batched_targets(
            stride,
            targets,
        )

        repeated_per_anchor_targets = self._repeat_for_anchors(
            len(anchor_info.boxes_wh),
            batched_targets,
        )

        # scale the anchor boxes
        # for the given feature shape
        scaled_anchor_boxes = [
            FeatureShape(
                width=a.width * 1 / stride,
                height=a.height * 1 / stride,
            )
            for a in anchor_info.boxes_wh
        ]

        scaled_anchor_boxes_tensor = torch.tensor(
            scaled_anchor_boxes,
            device=batched_targets.device,
            dtype=batched_targets.dtype,
        )

        # matching
        # using wh ratio
        filtered_targets = self._filter_targets(
            scaled_anchor_boxes_tensor,
            repeated_per_anchor_targets,
        )

        filtered_targets, offsets = self._incorporate_neighbouring_cells(
            input_image_shape,
            stride,
            filtered_targets,
        )

        sample_ids, labels = filtered_targets[:, :2].long().T
        cxcy = filtered_targets[:, 2:4]
        wh = filtered_targets[:, 4:6]

        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = (cxcy - offsets).long()
        # Isolate x and y index dimensions
        # grid xy indices
        gi, gj = gij.T
        anchor_indices = filtered_targets[:, 6].long()

        target_feature_shape = FeatureShape(
            width=input_image_shape.width // stride,
            height=input_image_shape.height // stride,
        )

        indices = AssignmentTargetIndices(
            samples=sample_ids,
            anchors=anchor_indices,
            grid_y=gj.clamp(0, target_feature_shape.height - 1).long(),
            grid_x=gi.clamp(0, target_feature_shape.width - 1).long(),
        )

        # gt_boxes = filtered_targets[:, 2:6]

        # cxcy = filtered_targets[:, 2:4]

        gt_boxes = torch.cat((cxcy - gij, wh), 1)

        # to xyxy
        # gt_boxes = tv.ops.box_convert(
        #     gt_boxes,
        #     in_fmt="cxcywh",
        #     out_fmt="xyxy",
        # )

        return AssignmentTargetInfo(
            indices=indices,
            gt_boxes=gt_boxes,
            labels=labels,
            anchors=scaled_anchor_boxes_tensor[anchor_indices],
            target_feature_shape=target_feature_shape,
        )

    def __call__(
        self,
        input_image_shape: FeatureShape,
        targets: Sequence[DetectionTarget],
    ) -> AssignmentResult:
        # input image shape and targets is
        # same for all layers
        assign_fn = partial(
            self._assign_per_layer,
            input_image_shape=input_image_shape,
            targets=targets,
        )

        ll_target_info = assign_fn(anchor_info=self.anchor_info.ll)
        ml_target_info = assign_fn(anchor_info=self.anchor_info.ml)
        hl_target_info = assign_fn(anchor_info=self.anchor_info.hl)

        return AssignmentResult(
            ll=ll_target_info,
            ml=ml_target_info,
            hl=hl_target_info,
        )
