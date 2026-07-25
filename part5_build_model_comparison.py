import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODEL_ORDER = {
    "Baseline-CustomCNN": 0,
    "MobileNetV2-Partial": 1,
    "mobilenetv2_partial": 1,
    "ResNet18-Layer4": 2,
}
MODEL_DISPLAY_NAMES = {
    "mobilenetv2_partial": "MobileNetV2-Partial",
}


def load_analysis_summaries(analysis_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for summary_path in analysis_dir.glob("*/analysis_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["_directory"] = str(summary_path.parent)
        summaries.append(summary)
    summaries.sort(
        key=lambda summary: (
            MODEL_ORDER.get(str(summary["model"]), len(MODEL_ORDER)),
            str(summary["model"]),
        )
    )
    if not summaries:
        raise FileNotFoundError(f"No model analysis summaries found in {analysis_dir}")
    return summaries


def comparison_rows(
    summaries: list[dict[str, Any]],
    reported_baseline_summary: Path | None = None,
) -> list[dict[str, Any]]:
    baseline_reported_accuracy = None
    if reported_baseline_summary is not None:
        original = json.loads(reported_baseline_summary.read_text(encoding="utf-8"))
        baseline_reported_accuracy = float(original["test_acc"])

    rows = []
    for summary in summaries:
        model_name = MODEL_DISPLAY_NAMES.get(
            str(summary["model"]),
            str(summary["model"]),
        )
        metrics = summary["metrics"]
        source_metrics = summary["source_metrics"]
        lab_accuracy = float(source_metrics["lab"]["accuracy"])
        field_accuracy = float(source_metrics["field"]["accuracy"])
        recomputed_accuracy = float(metrics["accuracy"])
        reported_accuracy = recomputed_accuracy
        if model_name == "Baseline-CustomCNN":
            reported_accuracy = baseline_reported_accuracy or recomputed_accuracy
        rows.append(
            {
                "model": model_name,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "num_samples": int(metrics["num_samples"]),
                "accuracy": recomputed_accuracy,
                "owner_reported_accuracy": reported_accuracy,
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "top5_accuracy": float(metrics["top5_accuracy"]),
                "lab_accuracy": lab_accuracy,
                "field_accuracy": field_accuracy,
                "lab_field_accuracy_gap": lab_accuracy - field_accuracy,
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def combine_per_class_metrics(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        model_name = MODEL_DISPLAY_NAMES.get(
            str(summary["model"]),
            str(summary["model"]),
        )
        source_path = Path(summary["_directory"]) / "per_class_metrics.csv"
        with source_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                rows.append({"model": model_name, **row})
    return rows


def plot_model_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Accuracy", "Macro-F1", "Weighted-F1", "Top-5"]
    keys = ["accuracy", "macro_f1", "weighted_f1", "top5_accuracy"]
    colors = ["#2A6F97", "#40916C", "#D97706"]
    x_values = np.arange(len(labels))
    width = 0.24

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for model_index, row in enumerate(rows):
        offset = (model_index - (len(rows) - 1) / 2) * width
        values = [float(row[key]) for key in keys]
        bars = axis.bar(
            x_values + offset,
            values,
            width,
            label=str(row["model"]),
            color=colors[model_index % len(colors)],
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    axis.set_title("Locked Test-Set Model Comparison")
    axis.set_ylabel("Score")
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0.75, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_source_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["model"]) for row in rows]
    lab_values = [float(row["lab_accuracy"]) for row in rows]
    field_values = [float(row["field_accuracy"]) for row in rows]
    x_values = np.arange(len(labels))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9, 5.5))
    lab_bars = axis.bar(
        x_values - width / 2,
        lab_values,
        width,
        label="Lab",
        color="#2A6F97",
    )
    field_bars = axis.bar(
        x_values + width / 2,
        field_values,
        width,
        label="Field",
        color="#D97706",
    )
    axis.bar_label(lab_bars, fmt="%.3f", padding=2, fontsize=9)
    axis.bar_label(field_bars, fmt="%.3f", padding=2, fontsize=9)
    axis.set_title("Lab vs Field Test Accuracy")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0.65, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_comparison(
    analysis_dir: Path,
    output_dir: Path,
    reported_baseline_summary: Path | None = None,
) -> list[dict[str, Any]]:
    summaries = load_analysis_summaries(analysis_dir)
    rows = comparison_rows(summaries, reported_baseline_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "model_comparison.csv", rows)
    save_csv(
        output_dir / "source_comparison.csv",
        [
            {
                "model": row["model"],
                "lab_accuracy": row["lab_accuracy"],
                "field_accuracy": row["field_accuracy"],
                "lab_field_accuracy_gap": row["lab_field_accuracy_gap"],
            }
            for row in rows
        ],
    )
    save_csv(
        output_dir / "per_class_metrics.csv",
        combine_per_class_metrics(summaries),
    )
    plot_model_comparison(output_dir / "model_comparison.png", rows)
    plot_source_comparison(output_dir / "source_comparison.png", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final comparison for all analysed models."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison"),
    )
    parser.add_argument("--reported-baseline-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_comparison(
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        reported_baseline_summary=args.reported_baseline_summary,
    )
    print(json.dumps(rows, indent=2))
    print(f"Saved comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
