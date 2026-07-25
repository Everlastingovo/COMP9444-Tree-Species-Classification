from collections.abc import Sequence
from typing import Any

import torch

from src.evaluation.confusion import confusion_matrix


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    _validate_logits_and_targets(logits, targets)
    predictions = logits.argmax(dim=1)
    return (predictions == targets).to(torch.float64).mean().item()


def top_k_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    _validate_logits_and_targets(logits, targets)
    if k <= 0:
        raise ValueError("k must be positive.")

    actual_k = min(k, logits.shape[1])
    top_k = logits.topk(actual_k, dim=1).indices
    correct = (top_k == targets.unsqueeze(1)).any(dim=1)
    return correct.to(torch.float64).mean().item()


def per_class_metrics_from_confusion(
    matrix: torch.Tensor,
    class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    _validate_confusion_matrix(matrix)
    num_classes = matrix.shape[0]
    if class_names is None:
        class_names = [str(index) for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("class_names must match the confusion matrix size.")

    values = matrix.to(torch.float64)
    true_positives = values.diag()
    support = values.sum(dim=1)
    predicted = values.sum(dim=0)

    precision = _safe_divide(true_positives, predicted)
    recall = _safe_divide(true_positives, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(class_names):
        rows.append(
            {
                "class_idx": index,
                "class_name": class_name,
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "f1": f1[index].item(),
                "support": int(support[index].item()),
            }
        )
    return rows


def aggregate_metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float | int]:
    _validate_confusion_matrix(matrix)
    values = matrix.to(torch.float64)
    support = values.sum(dim=1)
    total = support.sum()
    if total.item() == 0:
        raise ValueError("confusion matrix must contain at least one sample.")

    rows = per_class_metrics_from_confusion(matrix)
    precision = torch.tensor([row["precision"] for row in rows], dtype=torch.float64)
    recall = torch.tensor([row["recall"] for row in rows], dtype=torch.float64)
    f1 = torch.tensor([row["f1"] for row in rows], dtype=torch.float64)
    supported = support > 0
    weights = support / total

    return {
        "num_samples": int(total.item()),
        "num_classes": int(matrix.shape[0]),
        "classes_with_support": int(supported.sum().item()),
        "accuracy": values.diag().sum().item() / total.item(),
        "macro_precision": precision[supported].mean().item(),
        "macro_recall": recall[supported].mean().item(),
        "macro_f1": f1[supported].mean().item(),
        "weighted_precision": (precision * weights).sum().item(),
        "weighted_recall": (recall * weights).sum().item(),
        "weighted_f1": (f1 * weights).sum().item(),
    }


def classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]], torch.Tensor]:
    """Calculate the common project metrics from one set of model outputs."""
    _validate_logits_and_targets(logits, targets)
    num_classes = logits.shape[1]
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError("class_names must match the number of model outputs.")

    predictions = logits.argmax(dim=1)
    matrix = confusion_matrix(predictions, targets, num_classes)
    metrics = aggregate_metrics_from_confusion(matrix)
    metrics["top5_accuracy"] = top_k_accuracy_from_logits(logits, targets, k=5)
    per_class = per_class_metrics_from_confusion(matrix, class_names)
    return metrics, per_class, matrix


def metrics_by_source(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sources: Sequence[str],
    class_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Calculate the same metrics separately for lab and field images."""
    _validate_logits_and_targets(logits, targets)
    if len(sources) != targets.numel():
        raise ValueError("sources must contain one value for every target.")

    result: dict[str, dict[str, float | int]] = {}
    for source in sorted(set(sources)):
        mask = torch.tensor([value == source for value in sources], dtype=torch.bool)
        if not mask.any():
            continue
        metrics, _, _ = classification_metrics_from_logits(
            logits[mask],
            targets[mask],
            class_names,
        )
        result[source] = metrics
    return result


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [images, classes].")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [images].")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must contain the same number of images.")
    if logits.shape[0] == 0:
        raise ValueError("logits and targets cannot be empty.")
    if logits.shape[1] <= 1:
        raise ValueError("logits must contain at least two classes.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values.")

    targets_on_cpu = targets.detach().to(torch.int64).cpu()
    if targets_on_cpu.min().item() < 0 or targets_on_cpu.max().item() >= logits.shape[1]:
        raise ValueError("targets contain a class index outside the model output range.")


def _validate_confusion_matrix(matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square.")
    if matrix.shape[0] == 0:
        raise ValueError("confusion matrix cannot be empty.")
    if torch.any(matrix < 0):
        raise ValueError("confusion matrix cannot contain negative values.")
