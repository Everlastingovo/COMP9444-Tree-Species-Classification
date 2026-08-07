import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from src.evaluation.confusion import (
    confusion_matrix,
    plot_confusion_matrix,
    save_confusion_csv,
    strongest_confusion_pairs,
)
from src.evaluation.error_analysis import (
    low_confidence_correct,
    plot_error_cases,
    save_error_rows,
    top_errors,
)
from src.evaluation.metrics import (
    aggregate_metrics_from_confusion,
    per_class_metrics_from_confusion,
)


STANDARD_FIELDS = [
    "path",
    "source",
    "true_idx",
    "true_label",
    "predicted_idx",
    "predicted_label",
    "confidence",
    "correct",
    "top5_labels",
    "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_class_mapping(path: Path) -> dict[str, int]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    result = {str(label): int(index) for label, index in mapping.items()}
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError("class mapping must use contiguous indices starting at zero.")
    return result


def ordered_class_names(class_to_idx: dict[str, int]) -> list[str]:
    names = [""] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        names[class_index] = class_name
    return names


def standardize_prediction_rows(
    rows: list[dict[str, str]],
    class_to_idx: dict[str, int],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("prediction CSV cannot be empty.")

    class_names = ordered_class_names(class_to_idx)
    standardized = []
    seen_paths: set[str] = set()
    for row_index, row in enumerate(rows):
        path = pick(row, "path", "image_path")
        source = pick(row, "source")
        true_index = int(pick(row, "true_idx", "true_class_index"))
        true_label = pick(row, "true_label", "true_class_name")
        predicted_index = int(
            pick(row, "predicted_idx", "predicted_class_index")
        )
        predicted_label = pick(
            row,
            "predicted_label",
            "predicted_class_name",
        )
        confidence = float(pick(row, "confidence"))
        top5_labels = parse_top5_labels(row)
        correct = true_index == predicted_index

        if path in seen_paths:
            raise ValueError(f"duplicate prediction path: {path}")
        seen_paths.add(path)
        if source not in {"lab", "field"}:
            raise ValueError(f"row {row_index} has invalid source: {source}")
        if not 0 <= true_index < len(class_names):
            raise ValueError(f"row {row_index} has invalid true class index.")
        if not 0 <= predicted_index < len(class_names):
            raise ValueError(f"row {row_index} has invalid predicted class index.")
        if class_names[true_index] != true_label:
            raise ValueError(f"row {row_index} true class name does not match its index.")
        if class_names[predicted_index] != predicted_label:
            raise ValueError(
                f"row {row_index} predicted class name does not match its index."
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"row {row_index} confidence is outside [0, 1].")
        if "correct" in row and parse_bool(row["correct"]) != correct:
            raise ValueError(f"row {row_index} has an inconsistent correct value.")

        standardized.append(
            {
                "path": path,
                "source": source,
                "true_idx": true_index,
                "true_label": true_label,
                "predicted_idx": predicted_index,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": correct,
                "top5_labels": "|".join(top5_labels),
                "reason": "",
            }
        )
    return standardized


def calculate_metrics(
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> tuple[dict[str, float | int], list[dict[str, Any]], torch.Tensor]:
    targets = torch.tensor([int(row["true_idx"]) for row in rows])
    predictions = torch.tensor([int(row["predicted_idx"]) for row in rows])
    matrix = confusion_matrix(predictions, targets, len(class_names))
    metrics = aggregate_metrics_from_confusion(matrix)
    metrics["top5_accuracy"] = top5_accuracy_from_rows(rows)
    per_class = per_class_metrics_from_confusion(matrix, class_names)
    return metrics, per_class, matrix


def calculate_source_metrics(
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    result = []
    for source in ("lab", "field"):
        source_rows = [row for row in rows if row["source"] == source]
        if not source_rows:
            continue
        metrics, _, _ = calculate_metrics(source_rows, class_names)
        result.append({"source": source, **metrics})
    return result


def top5_accuracy_from_rows(rows: list[dict[str, Any]]) -> float:
    if any(not row["top5_labels"] for row in rows):
        raise ValueError("every prediction row must include Top-5 labels.")
    correct = sum(
        str(row["true_label"]) in str(row["top5_labels"]).split("|")
        for row in rows
    )
    return correct / len(rows)


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse_predictions(
    prediction_path: Path,
    class_mapping_path: Path,
    output_dir: Path,
    model_name: str,
    checkpoint_sha256: str = "",
    error_limit: int = 30,
    data_root: Path | None = None,
) -> dict[str, Any]:
    class_to_idx = load_class_mapping(class_mapping_path)
    class_names = ordered_class_names(class_to_idx)
    rows = standardize_prediction_rows(read_csv(prediction_path), class_to_idx)
    metrics, per_class, matrix = calculate_metrics(rows, class_names)
    source_metrics = calculate_source_metrics(rows, class_names)
    confusion_pairs = strongest_confusion_pairs(matrix, class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows(output_dir / "predictions_standardized.csv", rows)
    save_rows(output_dir / "per_class_metrics.csv", per_class)
    save_rows(output_dir / "source_metrics.csv", source_metrics)
    save_rows(output_dir / "strongest_confusion_pairs.csv", confusion_pairs)
    error_rows = top_errors(rows, limit=error_limit)
    uncertain_correct_rows = low_confidence_correct(rows, limit=error_limit)
    save_error_rows(output_dir / "high_confidence_errors.csv", error_rows)
    save_error_rows(
        output_dir / "low_confidence_correct.csv",
        uncertain_correct_rows,
    )
    if data_root is not None and error_rows:
        plot_error_cases(
            output_dir / "error_cases.png",
            error_rows,
            data_root=data_root,
            limit=min(12, len(error_rows)),
            title=f"{model_name} - High-Confidence Test Errors",
        )
    save_confusion_csv(output_dir / "confusion_matrix.csv", matrix, class_names)
    plot_confusion_matrix(
        output_dir / "confusion_matrix_raw.png",
        matrix,
        class_names,
        title=f"{model_name} - Test Confusion Matrix",
    )
    plot_confusion_matrix(
        output_dir / "confusion_matrix_normalized.png",
        matrix,
        class_names,
        normalize=True,
        title=f"{model_name} - Normalized Test Confusion Matrix",
    )

    summary = {
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_file": str(prediction_path),
        "class_mapping_file": str(class_mapping_path),
        "metrics": metrics,
        "source_metrics": {
            str(row["source"]): {
                key: value for key, value in row.items() if key != "source"
            }
            for row in source_metrics
        },
        "weakest_classes_by_f1": sorted(
            per_class,
            key=lambda row: (float(row["f1"]), str(row["class_name"])),
        )[:5],
        "strongest_confusion_pairs": confusion_pairs,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise ValueError(f"prediction row is missing one of these fields: {names}")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def parse_top5_labels(row: dict[str, str]) -> list[str]:
    if row.get("top5_labels"):
        return [value for value in row["top5_labels"].split("|") if value]
    if row.get("top5_class_names"):
        raw_value = row["top5_class_names"]
        if raw_value.lstrip().startswith("["):
            values = json.loads(raw_value)
            if not isinstance(values, list):
                raise ValueError("top5_class_names must contain a JSON list.")
            return [str(value) for value in values]
        return [value for value in raw_value.split("|") if value]
    raise ValueError("prediction row does not include Top-5 labels.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize and analyse saved Part 5 prediction files."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/metadata/class_to_idx.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--error-limit", type=int, default=30)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional raw-data root used to render the error-case image grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyse_predictions(
        prediction_path=args.predictions,
        class_mapping_path=args.class_mapping,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_sha256=args.checkpoint_sha256,
        error_limit=args.error_limit,
        data_root=args.data_root,
    )
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
