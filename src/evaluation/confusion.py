import csv
from pathlib import Path
from typing import Literal, Sequence

import torch


NormalizationMode = Literal["true", "predicted", "all"]


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build a matrix whose rows are true classes and columns are predictions."""
    predictions = predictions.detach().to(torch.int64).view(-1).cpu()
    targets = targets.detach().to(torch.int64).view(-1).cpu()

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if predictions.numel() != targets.numel():
        raise ValueError("predictions and targets must contain the same number of values.")
    if predictions.numel() == 0:
        raise ValueError("predictions and targets cannot be empty.")
    if predictions.min().item() < 0 or predictions.max().item() >= num_classes:
        raise ValueError("predictions contain a class index outside the valid range.")
    if targets.min().item() < 0 or targets.max().item() >= num_classes:
        raise ValueError("targets contain a class index outside the valid range.")

    encoded = targets * num_classes + predictions
    return torch.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes,
        num_classes,
    )


def normalize_confusion_matrix(
    matrix: torch.Tensor,
    mode: NormalizationMode = "true",
) -> torch.Tensor:
    """Normalize by true class, predicted class, or the whole matrix."""
    _validate_square_matrix(matrix)
    values = matrix.to(torch.float64)

    if mode == "true":
        denominator = values.sum(dim=1, keepdim=True)
    elif mode == "predicted":
        denominator = values.sum(dim=0, keepdim=True)
    elif mode == "all":
        denominator = values.sum().reshape(1, 1)
    else:
        raise ValueError("mode must be 'true', 'predicted', or 'all'.")

    return torch.where(
        denominator > 0,
        values / denominator,
        torch.zeros_like(values),
    )


def strongest_confusion_pairs(
    matrix: torch.Tensor,
    class_names: Sequence[str],
    limit: int = 10,
) -> list[dict[str, int | str]]:
    """Return the class pairs with the most mistakes in either direction."""
    _validate_square_matrix(matrix)
    if len(class_names) != matrix.shape[0]:
        raise ValueError("class_names must match the confusion matrix size.")
    if limit < 0:
        raise ValueError("limit cannot be negative.")

    pairs: list[dict[str, int | str]] = []
    for first_index in range(len(class_names)):
        for second_index in range(first_index + 1, len(class_names)):
            first_as_second = int(matrix[first_index, second_index].item())
            second_as_first = int(matrix[second_index, first_index].item())
            total = first_as_second + second_as_first
            if total == 0:
                continue
            pairs.append(
                {
                    "class_a": class_names[first_index],
                    "class_b": class_names[second_index],
                    "a_predicted_as_b": first_as_second,
                    "b_predicted_as_a": second_as_first,
                    "total_confusions": total,
                }
            )

    return sorted(
        pairs,
        key=lambda row: (
            -int(row["total_confusions"]),
            str(row["class_a"]),
            str(row["class_b"]),
        ),
    )[:limit]


def save_confusion_csv(
    path: Path,
    matrix: torch.Tensor,
    class_names: Sequence[str],
) -> None:
    _validate_square_matrix(matrix)
    if len(class_names) != matrix.shape[0]:
        raise ValueError("class_names must match the confusion matrix size.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_class", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def plot_confusion_matrix(
    path: Path,
    matrix: torch.Tensor,
    class_names: Sequence[str],
    *,
    normalize: bool = False,
    title: str | None = None,
) -> None:
    """Save a readable raw-count or true-class-normalized confusion matrix."""
    _validate_square_matrix(matrix)
    if len(class_names) != matrix.shape[0]:
        raise ValueError("class_names must match the confusion matrix size.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = normalize_confusion_matrix(matrix, "true") if normalize else matrix
    color_label = "Proportion" if normalize else "Images"
    default_title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    figure_size = max(10.0, len(class_names) * 0.45)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(values.numpy(), interpolation="nearest", cmap="Blues")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(color_label)
    axis.set(
        title=title or default_title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _validate_square_matrix(matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square.")
    if matrix.shape[0] == 0:
        raise ValueError("confusion matrix cannot be empty.")
    if torch.any(matrix < 0):
        raise ValueError("confusion matrix cannot contain negative values.")
