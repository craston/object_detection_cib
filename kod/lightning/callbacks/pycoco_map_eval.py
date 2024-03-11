from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence
from typing import NamedTuple

import torch

from rich.table import Table
from rich.console import Console

import lightning.pytorch as LP
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from vision_evaluation.evaluators import CocoMeanAveragePrecisionEvaluator

from kod.data.detection import DetectionTarget

from .map_eval import MAPEvalCallbackInput


class PyCocoGroundTruth(NamedTuple):
    label_id: float
    l: float
    t: float
    r: float
    b: float


class PyCocoTarget(NamedTuple):
    label_id: float
    score: float
    l: float
    t: float
    r: float
    b: float


class PyCOCOMAPEvalCallback(Callback):
    def __init__(self, label_info: dict[int, str], **kwargs):
        super().__init__(**kwargs)
        self.label_info = label_info
        self.evaluator = CocoMeanAveragePrecisionEvaluator(
            ious=[0.3, 0.5, 0.75, 0.9],
            report_tag_wise=[False, True, False, False],
        )

    def on_validation_batch_end(
        self,
        trainer: LP.Trainer,
        pl_module: LP.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(outputs, dict)

        expected_input: MAPEvalCallbackInput = outputs  # type: ignore

        targets: Sequence[DetectionTarget] = expected_input["targets"]
        detections: Sequence[torch.Tensor] = expected_input["detections"]

        prediction_results: list[list[PyCocoTarget]] = []
        for idx in range(len(targets)):
            prediction_results.append([])

            fd = detections[idx].cpu()

            for fd_idx in range(len(fd)):
                p = fd[fd_idx].tolist()
                prediction_results[idx].append(
                    PyCocoTarget(
                        label_id=p[5],
                        score=p[4],
                        l=p[0],
                        t=p[1],
                        r=p[2],
                        b=p[3],
                    )
                )

        gt_targets: list[list[PyCocoGroundTruth]] = []
        for idx in range(len(targets)):
            gt_targets.append([])

            boxes = targets[idx].boxes.cpu()
            labels = targets[idx].labels.cpu()

            for box_idx in range(len(boxes)):
                b = boxes[box_idx].tolist()
                label_id = labels[box_idx].item()

                gt_targets[idx].append(
                    PyCocoGroundTruth(
                        label_id=label_id,
                        l=b[0],
                        t=b[1],
                        r=b[2],
                        b=b[3],
                    )
                )

        self.evaluator.add_predictions(prediction_results, gt_targets)

    def on_validation_epoch_end(
        self,
        trainer: LP.Trainer,
        pl_module: LP.LightningModule,
    ) -> None:
        report = self.evaluator.get_report()

        results = {
            "map": report["avg_mAP"],
            "map30": report["mAP_30"],
            "map50": report["mAP_50"],
            "map75": report["mAP_75"],
            "map90": report["mAP_90"],
        }

        for k, v in report["tag_wise_AP_50"].items():
            cls_name = self.label_info[k]
            results[f"map50_{cls_name}"] = v

        table = Table(
            title=f"MAP [Epoch - {trainer.current_epoch}]",
            show_header=False,
            show_lines=True,
        )

        for k, v in results.items():
            table.add_row(k, str(v))

        console = Console()
        console.print(table)

        pl_module.log_dict(
            results,
            sync_dist=True,
        )

        self.evaluator.reset()
