import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def prediction_rows_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    *,
    image_paths: Sequence[str | Path] | None = None,
    sources: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Create traceable per-image rows for later error and source analysis."""
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits must be 2D and targets must be 1D.")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must contain the same number of images.")
    if logits.shape[1] != len(class_names):
        raise ValueError("class_names must match the number of model outputs.")
    if image_paths is not None and len(image_paths) != targets.numel():
        raise ValueError("image_paths must contain one path for every target.")
    if sources is not None and len(sources) != targets.numel():
        raise ValueError("sources must contain one value for every target.")

    probabilities = torch.softmax(logits.detach(), dim=1).cpu()
    targets = targets.detach().to(torch.int64).cpu()
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    top_k = min(5, logits.shape[1])
    top_indices = probabilities.topk(top_k, dim=1).indices

    rows: list[dict[str, Any]] = []
    for index in range(targets.numel()):
        true_index = int(targets[index].item())
        predicted_index = int(predictions[index].item())
        rows.append(
            {
                "image_index": index,
                "path": str(image_paths[index]) if image_paths is not None else "",
                "source": sources[index] if sources is not None else "",
                "true_idx": true_index,
                "true_label": class_names[true_index],
                "predicted_idx": predicted_index,
                "predicted_label": class_names[predicted_index],
                "confidence": confidences[index].item(),
                "correct": true_index == predicted_index,
                "top5_labels": "|".join(
                    class_names[int(class_index)]
                    for class_index in top_indices[index].tolist()
                ),
                "reason": "",
            }
        )
    return rows


def top_errors(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Return the highest-confidence incorrect predictions first."""
    _validate_limit(limit)
    errors = [row for row in rows if not _row_is_correct(row)]
    return sorted(
        errors,
        key=lambda row: (-float(row.get("confidence", 0.0)), _row_path(row)),
    )[:limit]


def low_confidence_correct(
    rows: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return correct predictions that the model was least certain about."""
    _validate_limit(limit)
    correct_rows = [row for row in rows if _row_is_correct(row)]
    return sorted(
        correct_rows,
        key=lambda row: (float(row.get("confidence", 1.0)), _row_path(row)),
    )[:limit]


def errors_for_confusion_pair(
    rows: list[dict[str, Any]],
    class_a: str,
    class_b: str,
) -> list[dict[str, Any]]:
    """Return mistakes in both directions for one pair of similar classes."""
    selected = []
    for row in rows:
        true_label = _true_label(row)
        predicted_label = _predicted_label(row)
        if (true_label, predicted_label) in {(class_a, class_b), (class_b, class_a)}:
            selected.append(row)
    return sorted(
        selected,
        key=lambda row: (-float(row.get("confidence", 0.0)), _row_path(row)),
    )


def save_error_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_error_cases(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    data_root: Path | None = None,
    limit: int = 12,
    columns: int = 4,
    title: str = "Representative Error Cases",
) -> None:
    """Save an image grid with true label, prediction, and confidence."""
    _validate_limit(limit)
    if columns <= 0:
        raise ValueError("columns must be positive.")
    selected = rows[:limit]
    if not selected:
        raise ValueError("at least one row is required to plot error cases.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_count = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(columns * 3.5, row_count * 3.5),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")

    for axis, row in zip(axes.flat, selected):
        image_path = Path(_row_path(row))
        if data_root is not None and not image_path.is_absolute():
            image_path = data_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Error-case image not found: {image_path}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            axis.imshow(image)
        confidence = float(row.get("confidence", 0.0))
        axis.set_title(
            f"True: {_true_label(row)}\n"
            f"Pred: {_predicted_label(row)} ({confidence:.1%})",
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _row_is_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        value = row["correct"]
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)
    return _true_label(row) == _predicted_label(row)


def _true_label(row: dict[str, Any]) -> str:
    return str(row.get("true_label", row.get("label", "")))


def _predicted_label(row: dict[str, Any]) -> str:
    return str(row.get("predicted_label", row.get("prediction", "")))


def _row_path(row: dict[str, Any]) -> str:
    return str(row.get("path", ""))


def _validate_limit(limit: int) -> None:
    if limit < 0:
        raise ValueError("limit cannot be negative.")
