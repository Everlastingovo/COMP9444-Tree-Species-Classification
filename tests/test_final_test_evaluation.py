import torch
from PIL import Image

from src.data.transforms import LeafImageTransform
from src.evaluation.final_test import (
    classification_report_from_confusion,
    confusion_from_predictions,
    metrics_from_outputs,
    strongest_confusion_pairs,
)


def test_final_test_metrics_and_classification_report() -> None:
    class_names = ["a", "b", "c"]
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 0, 2])
    top5_correct = torch.ones(6, dtype=torch.bool)
    losses = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    metrics, report, confusion = metrics_from_outputs(
        targets,
        predictions,
        top5_correct,
        losses,
        class_names,
    )

    assert confusion.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert metrics["images"] == 6
    assert metrics["top1"] == 4 / 6
    assert metrics["top5"] == 1.0
    assert metrics["classes_with_support"] == 3
    assert [row["support"] for row in report] == [2, 2, 2]
    assert report[1]["precision"] == 2 / 3
    assert report[1]["recall"] == 1.0


def test_confusion_helpers_report_unordered_pairs() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2])
    predictions = torch.tensor([1, 1, 0, 2, 2])
    confusion = confusion_from_predictions(targets, predictions, num_classes=3)
    report, macro = classification_report_from_confusion(confusion, ["a", "b", "c"])
    pairs = strongest_confusion_pairs(confusion, ["a", "b", "c"])

    assert len(report) == 3
    assert macro["classes_with_support"] == 3
    assert pairs[0] == {
        "class_a": "a",
        "class_b": "b",
        "a_predicted_as_b": 2,
        "b_predicted_as_a": 1,
        "total_confusions": 3,
    }


def test_final_test_transform_is_deterministic_imagenet_preprocessing() -> None:
    transform = LeafImageTransform(
        image_size=224,
        training=False,
        augment=False,
        normalization="imagenet",
    )
    image = Image.new("RGB", (180, 260), (50, 100, 150))

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)
    assert transform.training is False
    assert transform.augment is False
    assert transform.normalization == "imagenet"
