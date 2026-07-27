"""Build isolated Part 4 reselection tables and figures from saved outputs.

This script performs no model training and does not load any dataset images.
It reads only the saved histories and evaluation files under
``reselection_20260727_run1``.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_ROOT = Path("reselection_20260727_run1")
OUTPUT_ROOT = RUN_ROOT / "outputs"
TABLE_DIR = Path("report/tables")
FIGURE_DIR = Path("report/figures/reselection_20260727_run1")

RESNET_FROZEN_HISTORY = OUTPUT_ROOT / "resnet18/frozen/history.csv"
RESNET_FROZEN_SUMMARY = OUTPUT_ROOT / "resnet18/frozen/summary.json"
RESNET_FINETUNE_HISTORY = OUTPUT_ROOT / "resnet18/layer4/history.csv"
RESNET_FINETUNE_SUMMARY = OUTPUT_ROOT / "resnet18/layer4/summary.json"
RESNET_VALIDATION = OUTPUT_ROOT / "resnet18/layer4/validation/metrics.json"

EFFICIENTNET_FROZEN_HISTORY = OUTPUT_ROOT / "efficientnet_b0/frozen/history.csv"
EFFICIENTNET_FROZEN_SUMMARY = OUTPUT_ROOT / "efficientnet_b0/frozen/summary.json"
EFFICIENTNET_FINETUNE_HISTORY = (
    OUTPUT_ROOT / "efficientnet_b0/finetune/history.csv"
)
EFFICIENTNET_FINETUNE_SUMMARY = (
    OUTPUT_ROOT / "efficientnet_b0/finetune/summary.json"
)
EFFICIENTNET_VALIDATION = (
    OUTPUT_ROOT / "efficientnet_b0/finetune/validation/metrics.json"
)
EFFICIENTNET_TEST = OUTPUT_ROOT / "efficientnet_b0/finetune/test/metrics.json"
EFFICIENTNET_TEST_PER_CLASS = (
    OUTPUT_ROOT / "efficientnet_b0/finetune/test/per_class_metrics.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def best_epoch(history: list[dict[str, str]]) -> int:
    return int(max(history, key=lambda row: float(row["val_accuracy"]))["epoch"])


def require_files() -> None:
    required = [
        RESNET_FROZEN_HISTORY,
        RESNET_FROZEN_SUMMARY,
        RESNET_FINETUNE_HISTORY,
        RESNET_FINETUNE_SUMMARY,
        RESNET_VALIDATION,
        EFFICIENTNET_FROZEN_HISTORY,
        EFFICIENTNET_FROZEN_SUMMARY,
        EFFICIENTNET_FINETUNE_HISTORY,
        EFFICIENTNET_FINETUNE_SUMMARY,
        EFFICIENTNET_VALIDATION,
        EFFICIENTNET_TEST,
        EFFICIENTNET_TEST_PER_CLASS,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing reselection result files:\n"
            + "\n".join(str(path) for path in missing)
        )


def build_stage_tables() -> tuple[list[dict], list[dict]]:
    resnet_frozen_history = read_csv(RESNET_FROZEN_HISTORY)
    resnet_finetune_history = read_csv(RESNET_FINETUNE_HISTORY)
    resnet_frozen_summary = read_json(RESNET_FROZEN_SUMMARY)
    resnet_finetune_summary = read_json(RESNET_FINETUNE_SUMMARY)
    resnet_validation = read_json(RESNET_VALIDATION)

    resnet_rows = [
        {
            "model": "ResNet18",
            "stage": "Frozen head",
            "trainable_scope": "head",
            "learning_rate": 0.001,
            "epochs": len(resnet_frozen_history),
            "best_epoch": best_epoch(resnet_frozen_history),
            "trainable_parameters": resnet_frozen_summary["trainable_parameters"],
            "val_accuracy": resnet_frozen_summary["best_val_accuracy"],
            "val_macro_f1": "",
            "selected_for_model_comparison": False,
        },
        {
            "model": "ResNet18",
            "stage": "Layer4 fine-tuning",
            "trainable_scope": "layer4",
            "learning_rate": 0.00003,
            "epochs": len(resnet_finetune_history),
            "best_epoch": best_epoch(resnet_finetune_history),
            "trainable_parameters": resnet_finetune_summary[
                "trainable_parameters"
            ],
            "val_accuracy": resnet_validation["accuracy"],
            "val_macro_f1": resnet_validation["macro_f1"],
            "selected_for_model_comparison": True,
        },
    ]

    efficientnet_frozen_history = read_csv(EFFICIENTNET_FROZEN_HISTORY)
    efficientnet_finetune_history = read_csv(EFFICIENTNET_FINETUNE_HISTORY)
    efficientnet_frozen_summary = read_json(EFFICIENTNET_FROZEN_SUMMARY)
    efficientnet_finetune_summary = read_json(EFFICIENTNET_FINETUNE_SUMMARY)
    efficientnet_validation = read_json(EFFICIENTNET_VALIDATION)

    efficientnet_rows = [
        {
            "model": "EfficientNet-B0",
            "stage": "Frozen head",
            "trainable_scope": "head",
            "learning_rate": 0.001,
            "epochs": len(efficientnet_frozen_history),
            "best_epoch": best_epoch(efficientnet_frozen_history),
            "trainable_parameters": efficientnet_frozen_summary[
                "trainable_parameters"
            ],
            "val_accuracy": efficientnet_frozen_summary["best_val_accuracy"],
            "val_macro_f1": "",
            "selected_for_model_comparison": False,
        },
        {
            "model": "EfficientNet-B0",
            "stage": "Last-block fine-tuning",
            "trainable_scope": "last_blocks",
            "learning_rate": 0.00003,
            "epochs": len(efficientnet_finetune_history),
            "best_epoch": best_epoch(efficientnet_finetune_history),
            "trainable_parameters": efficientnet_finetune_summary[
                "trainable_parameters"
            ],
            "val_accuracy": efficientnet_validation["accuracy"],
            "val_macro_f1": efficientnet_validation["macro_f1"],
            "selected_for_model_comparison": True,
        },
    ]

    write_csv(
        TABLE_DIR / "reselection_resnet18_experiments.csv",
        resnet_rows,
    )
    write_csv(
        TABLE_DIR / "reselection_efficientnet_b0_experiments.csv",
        efficientnet_rows,
    )
    return resnet_rows, efficientnet_rows


def build_model_selection_table() -> list[dict]:
    resnet_summary = read_json(RESNET_FINETUNE_SUMMARY)
    resnet_validation = read_json(RESNET_VALIDATION)
    efficientnet_summary = read_json(EFFICIENTNET_FINETUNE_SUMMARY)
    efficientnet_validation = read_json(EFFICIENTNET_VALIDATION)

    rows = [
        {
            "model": "ResNet18",
            "total_parameters": resnet_summary["total_parameters"],
            "trainable_parameters": resnet_summary["trainable_parameters"],
            "val_accuracy": resnet_validation["accuracy"],
            "val_macro_f1": resnet_validation["macro_f1"],
            "val_weighted_f1": resnet_validation["weighted_f1"],
            "val_top5_accuracy": resnet_validation["top5_accuracy"],
            "selected_final_model": False,
            "selection_basis": "validation metrics",
        },
        {
            "model": "EfficientNet-B0",
            "total_parameters": efficientnet_summary["total_parameters"],
            "trainable_parameters": efficientnet_summary["trainable_parameters"],
            "val_accuracy": efficientnet_validation["accuracy"],
            "val_macro_f1": efficientnet_validation["macro_f1"],
            "val_weighted_f1": efficientnet_validation["weighted_f1"],
            "val_top5_accuracy": efficientnet_validation["top5_accuracy"],
            "selected_final_model": True,
            "selection_basis": "validation metrics",
        },
    ]
    write_csv(
        TABLE_DIR / "reselection_model_validation_comparison.csv",
        rows,
    )
    return rows


def build_final_test_tables() -> None:
    test_metrics = read_json(EFFICIENTNET_TEST)
    metric_fields = [
        "num_samples",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "top5_accuracy",
        "split",
        "evaluation_scope",
        "checkpoint",
        "config",
        "test_set_evaluated",
    ]
    write_csv(
        TABLE_DIR / "reselection_efficientnet_b0_final_test_metrics.csv",
        [{field: test_metrics[field] for field in metric_fields}],
    )
    write_csv(
        TABLE_DIR / "reselection_efficientnet_b0_test_per_class_metrics.csv",
        read_csv(EFFICIENTNET_TEST_PER_CLASS),
    )


def plot_training_curves() -> None:
    experiments = [
        ("ResNet18 Layer4", RESNET_FINETUNE_HISTORY),
        ("EfficientNet-B0 last blocks", EFFICIENTNET_FINETUNE_HISTORY),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for row_index, (name, history_path) in enumerate(experiments):
        history = read_csv(history_path)
        epochs = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]
        train_accuracy = [float(row["train_accuracy"]) * 100 for row in history]
        val_accuracy = [float(row["val_accuracy"]) * 100 for row in history]

        loss_axis = axes[row_index, 0]
        loss_axis.plot(epochs, train_loss, marker="o", label="Train")
        loss_axis.plot(epochs, val_loss, marker="s", label="Validation")
        loss_axis.set_title(f"{name} — loss")
        loss_axis.set_ylabel("Cross-entropy loss")

        accuracy_axis = axes[row_index, 1]
        accuracy_axis.plot(epochs, train_accuracy, marker="o", label="Train")
        accuracy_axis.plot(epochs, val_accuracy, marker="s", label="Validation")
        accuracy_axis.set_title(f"{name} — accuracy")
        accuracy_axis.set_ylabel("Accuracy (%)")

        for axis in (loss_axis, accuracy_axis):
            axis.set_xlabel("Epoch")
            axis.set_xticks(epochs)
            axis.grid(alpha=0.25)
            axis.legend()

    fig.suptitle(
        "Reselection Run: Fine-tuning Curves",
        fontsize=16,
    )
    fig.savefig(
        FIGURE_DIR / "reselection_finetuning_curves.png",
        dpi=300,
    )
    plt.close(fig)


def plot_validation_comparison(rows: list[dict]) -> None:
    names = [str(row["model"]) for row in rows]
    metrics = [
        ("val_accuracy", "Accuracy"),
        ("val_macro_f1", "Macro-F1"),
        ("val_weighted_f1", "Weighted-F1"),
        ("val_top5_accuracy", "Top-5 accuracy"),
    ]
    x_positions = list(range(len(names)))
    width = 0.18

    fig, axis = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    for metric_index, (field, label) in enumerate(metrics):
        positions = [
            position + (metric_index - 1.5) * width
            for position in x_positions
        ]
        values = [float(row[field]) * 100 for row in rows]
        bars = axis.bar(positions, values, width=width, label=label)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    axis.set_title("Validation-Based Final Model Selection")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(x_positions, names)
    axis.set_ylim(96, 100.75)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, loc="upper center")
    axis.text(
        0.5,
        -0.15,
        "EfficientNet-B0 selected before final test evaluation",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
    )
    fig.savefig(
        FIGURE_DIR / "reselection_validation_model_comparison.png",
        dpi=300,
    )
    plt.close(fig)


def plot_test_per_class_f1() -> None:
    rows = read_csv(EFFICIENTNET_TEST_PER_CLASS)
    rows.sort(key=lambda row: float(row["f1"]))
    names = [row["class_name"].replace("_", " ") for row in rows]
    scores = [float(row["f1"]) * 100 for row in rows]
    colors = [
        "#d97706" if index < 5 else "#2563eb"
        for index in range(len(rows))
    ]

    fig, axis = plt.subplots(figsize=(10, 11), constrained_layout=True)
    bars = axis.barh(names, scores, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=7)
    axis.set_title("Selected EfficientNet-B0: Final Test F1 by Species")
    axis.set_xlabel("F1 score (%)")
    axis.set_xlim(0, 105)
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(
        FIGURE_DIR / "reselection_efficientnet_b0_test_per_class_f1.png",
        dpi=300,
    )
    plt.close(fig)


def main() -> None:
    require_files()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    build_stage_tables()
    comparison_rows = build_model_selection_table()
    build_final_test_tables()
    plot_training_curves()
    plot_validation_comparison(comparison_rows)
    plot_test_per_class_f1()

    print(f"Reselection tables: {TABLE_DIR}")
    print(f"Reselection figures: {FIGURE_DIR}")
    print("No model training or dataset evaluation was performed.")


if __name__ == "__main__":
    main()
