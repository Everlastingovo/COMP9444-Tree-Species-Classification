from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from part5_analyse_predictions import (
    calculate_metrics,
    standardize_prediction_rows,
)
from src.evaluation.confusion import (
    confusion_matrix,
    normalize_confusion_matrix,
    plot_confusion_matrix,
    save_confusion_csv,
    strongest_confusion_pairs,
)
from src.evaluation.error_analysis import (
    errors_for_confusion_pair,
    low_confidence_correct,
    prediction_rows_from_logits,
    top_errors,
)
from src.evaluation.gradcam import (
    GradCAM,
    find_last_conv_layer,
    overlay_heatmap,
    save_gradcam_grid,
)
from src.evaluation.metrics import (
    classification_metrics_from_logits,
    metrics_by_source,
    top_k_accuracy_from_logits,
)
from src.evaluation.preprocessing import preprocess_evaluation_image


class TinyImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(6, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def fixed_outputs() -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.tensor(
        [
            [8.0, 1.0, 0.0],
            [1.0, 7.0, 0.0],
            [0.0, 9.0, 1.0],
            [0.0, 8.0, 1.0],
            [7.0, 1.0, 0.0],
            [0.0, 1.0, 8.0],
        ]
    )
    return logits, targets


def test_unified_metrics_include_macro_weighted_and_top5() -> None:
    logits, targets = fixed_outputs()
    metrics, per_class, matrix = classification_metrics_from_logits(
        logits,
        targets,
        ["a", "b", "c"],
    )

    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert metrics["num_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics["weighted_f1"] == pytest.approx(metrics["macro_f1"])
    assert metrics["top5_accuracy"] == 1.0
    assert [row["support"] for row in per_class] == [2, 2, 2]


def test_top_k_uses_available_classes_when_fewer_than_five() -> None:
    logits, targets = fixed_outputs()
    assert top_k_accuracy_from_logits(logits, targets, k=5) == 1.0


def test_source_metrics_use_the_same_evaluation_function() -> None:
    logits, targets = fixed_outputs()
    result = metrics_by_source(
        logits,
        targets,
        ["lab", "lab", "lab", "field", "field", "field"],
        ["a", "b", "c"],
    )

    assert set(result) == {"field", "lab"}
    assert result["lab"]["num_samples"] == 3
    assert result["field"]["num_samples"] == 3


def test_confusion_normalization_pairs_and_csv(tmp_path: Path) -> None:
    predictions = torch.tensor([0, 1, 1, 1, 0, 2])
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    matrix = confusion_matrix(predictions, targets, num_classes=3)
    normalized = normalize_confusion_matrix(matrix)
    pairs = strongest_confusion_pairs(matrix, ["a", "b", "c"])

    assert normalized.sum(dim=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert pairs[0]["class_a"] == "a"
    assert pairs[0]["class_b"] == "b"
    assert pairs[0]["total_confusions"] == 1

    output_path = tmp_path / "confusion.csv"
    save_confusion_csv(output_path, matrix, ["a", "b", "c"])
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == (
        "true_class,a,b,c"
    )


def test_error_rows_are_sorted_by_confidence() -> None:
    logits, targets = fixed_outputs()
    rows = prediction_rows_from_logits(
        logits,
        targets,
        ["a", "b", "c"],
        image_paths=[f"image-{index}.jpg" for index in range(6)],
        sources=["lab", "lab", "lab", "field", "field", "field"],
    )
    errors = top_errors(rows)
    uncertain_correct = low_confidence_correct(rows, limit=2)
    pair_errors = errors_for_confusion_pair(rows, "a", "b")

    assert len(errors) == 2
    assert errors[0]["confidence"] >= errors[1]["confidence"]
    assert all(row["correct"] is False for row in errors)
    assert len(uncertain_correct) == 2
    assert all(row["correct"] is True for row in uncertain_correct)
    assert len(pair_errors) == 1
    assert pair_errors[0]["true_label"] == "a"
    assert pair_errors[0]["predicted_label"] == "b"


def test_gradcam_works_for_a_generic_convolutional_model() -> None:
    torch.manual_seed(42)
    model = TinyImageClassifier()
    images = torch.rand(2, 3, 24, 24)

    assert find_last_conv_layer(model) is model.features[2]
    with GradCAM(model) as gradcam:
        heatmaps, logits, classes = gradcam.generate(images)

    assert heatmaps.shape == (2, 24, 24)
    assert logits.shape == (2, 3)
    assert classes.shape == (2,)
    assert torch.isfinite(heatmaps).all()
    assert float(heatmaps.min()) >= 0.0
    assert float(heatmaps.max()) <= 1.0


def test_gradcam_supports_an_explicit_target_class() -> None:
    model = TinyImageClassifier()
    images = torch.rand(1, 3, 16, 16)

    with GradCAM(model) as gradcam:
        _, _, classes = gradcam.generate(images, class_indices=[2])

    assert classes.tolist() == [2]


def test_report_visuals_can_be_saved(tmp_path: Path) -> None:
    matrix = torch.tensor([[3, 1], [1, 3]])
    confusion_path = tmp_path / "confusion.png"
    plot_confusion_matrix(
        confusion_path,
        matrix,
        ["class_a", "class_b"],
        normalize=True,
    )

    image = Image.new("RGB", (20, 20), (120, 180, 80))
    heatmap = torch.linspace(0.0, 1.0, steps=400).reshape(20, 20)
    overlay = overlay_heatmap(image, heatmap)
    gradcam_path = tmp_path / "gradcam.png"
    save_gradcam_grid(
        gradcam_path,
        [image],
        heatmap.unsqueeze(0),
        ["class_a"],
        ["class_b"],
        [0.91],
    )

    assert confusion_path.is_file()
    assert confusion_path.stat().st_size > 0
    assert overlay.size == image.size
    assert gradcam_path.is_file()
    assert gradcam_path.stat().st_size > 0


def test_part3_handoff_columns_are_standardized() -> None:
    class_to_idx = {"class_a": 0, "class_b": 1}
    original_rows = [
        {
            "image_path": "dataset/images/lab/a.jpg",
            "true_class_index": "0",
            "true_class_name": "class_a",
            "predicted_class_index": "1",
            "predicted_class_name": "class_b",
            "confidence": "0.8",
            "correct": "false",
            "source": "lab",
            "top5_class_names": "[\"class_b\", \"class_a\"]",
        },
        {
            "image_path": "dataset/images/field/b.jpg",
            "true_class_index": "1",
            "true_class_name": "class_b",
            "predicted_class_index": "1",
            "predicted_class_name": "class_b",
            "confidence": "0.9",
            "correct": "true",
            "source": "field",
            "top5_class_names": "[\"class_b\", \"class_a\"]",
        },
    ]

    rows = standardize_prediction_rows(original_rows, class_to_idx)
    metrics, per_class, matrix = calculate_metrics(rows, ["class_a", "class_b"])

    assert rows[0]["path"] == "dataset/images/lab/a.jpg"
    assert rows[0]["top5_labels"] == "class_b|class_a"
    assert matrix.tolist() == [[0, 1], [0, 1]]
    assert metrics["accuracy"] == 0.5
    assert metrics["top5_accuracy"] == 1.0
    assert [row["support"] for row in per_class] == [1, 1]


def test_part4_pipe_separated_top5_columns_are_standardized() -> None:
    rows = standardize_prediction_rows(
        [
            {
                "path": "dataset/images/field/a.jpg",
                "source": "field",
                "true_class_index": "0",
                "true_class_name": "class_a",
                "predicted_class_index": "1",
                "predicted_class_name": "class_b",
                "confidence": "0.75",
                "correct": "False",
                "top5_class_names": "class_b|class_a",
            }
        ],
        {"class_a": 0, "class_b": 1},
    )

    assert rows[0]["top5_labels"] == "class_b|class_a"


def test_shared_evaluation_preprocessing_supports_brightness() -> None:
    image = Image.new("RGB", (30, 20), (120, 100, 80))
    original = preprocess_evaluation_image(
        image,
        image_size=24,
        normalization="standard",
    )
    darker = preprocess_evaluation_image(
        image,
        image_size=24,
        normalization="standard",
        brightness_factor=0.6,
    )

    assert original.shape == (3, 24, 24)
    assert torch.isfinite(original).all()
    assert darker.mean() < original.mean()


def test_shared_evaluation_preprocessing_validates_settings() -> None:
    image = Image.new("RGB", (10, 10), (120, 100, 80))
    with pytest.raises(ValueError, match="brightness_factor"):
        preprocess_evaluation_image(
            image,
            image_size=24,
            normalization="standard",
            brightness_factor=0.0,
        )
