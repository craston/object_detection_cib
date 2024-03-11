import fiftyone as fo
import matplotlib.pyplot as plt

from kod.data.enums import DatasetName
from kod.data.builder import _load_dataset
from kod.data.builder import _get_image_count


def plot_dataset_distribution(
    dataset_name: DatasetName, split: str, plot_title: str
) -> None:
    dataset_zipf: fo.Dataset = _load_dataset(dataset_name, split=split)

    zipf_counts: dict = dataset_zipf.count_values("ground_truth.detections.label")
    zipf_counts = dict(sorted(zipf_counts.items(), key=lambda x: x[1], reverse=True))
    print("instance counts: ", zipf_counts)

    plt.figure()
    plt.bar(zipf_counts.keys(), zipf_counts.values())
    plt.xticks(rotation=45)
    plt.title(f"{dataset_name.value}")
    plt.ylabel("Number of instances")
    plt.tight_layout()

    plt.savefig(f"{plot_title}_instances.png")

    image_zipf_count = _get_image_count(dataset_zipf, dataset_zipf.default_classes)
    image_zipf_count = dict(
        sorted(image_zipf_count.items(), key=lambda x: x[1], reverse=True)
    )
    print("image counts: ", image_zipf_count)
    plt.figure()
    plt.bar(image_zipf_count.keys(), image_zipf_count.values())
    plt.xticks(rotation=45)
    plt.title(f"{dataset_name.value}")
    plt.ylabel("Number of images")
    plt.tight_layout()

    plt.savefig("{plot_title}_images.png")


plot_dataset_distribution(DatasetName.oi_zipf, "train", "oi_zipf")
