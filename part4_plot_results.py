import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = [
    {
        "name": "Frozen head",
        "scope": "head",
        "learning_rate": 1e-3,
        "history": Path("outputs/resnet18/frozen/history.csv"),
        "validation_metrics": Path("outputs/resnet18/frozen/validation/metrics.json"),
    },
    {
        "name": "Layer4 (LR=1e-4)",
        "scope": "layer4",
        "learning_rate": 1e-4,
        "history": Path("outputs/resnet18/layer4/history.csv"),
        "validation_metrics": Path("outputs/resnet18/layer4/validation/metrics.json"),
    },
    {
        "name": "Layer4 (LR=3e-5)",
        "scope": "layer4",
        "learning_rate": 3e-5,
        "history": Path("outputs/resnet18/layer4_lr3e5/history.csv"),
        "validation_metrics": Path(
            "outputs/resnet18/layer4_lr3e5/validation/metrics.json"
        ),
        "test_metrics": Path("outputs/resnet18/layer4_lr3e5/test/metrics.json"),
    },
]

EFFICIENTNET_EXPERIMENTS = [
    {
        "name": "EfficientNet-B0 frozen head",
        "scope": "head",
        "learning_rate": 1e-3,
        "history": Path("outputs/efficientnet_b0/frozen/history.csv"),
    },
    {
        "name": "EfficientNet-B0 fine-tuned",
        "scope": "last_blocks",
        "learning_rate": 3e-5,
        "history": Path("outputs/efficientnet_b0/finetune/history.csv"),
        "validation_metrics": Path(
            "outputs/efficientnet_b0/finetune/validation/metrics.json"
        ),
    },
]

FIGURE_DIR = Path("report/figures/resnet18")
EFFICIENTNET_FIGURE_DIR = Path("report/figures/efficientnet_b0")
TABLE_DIR = Path("report/tables")
OUTPUT_DIR = Path("outputs/resnet18")
TEST_PER_CLASS_PATH = Path(
    "outputs/resnet18/layer4_lr3e5/test/per_class_metrics.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_files() -> None:
    required = [TEST_PER_CLASS_PATH]
    for experiment in EXPERIMENTS:
        required.extend(
            [experiment["history"], experiment["validation_metrics"]]
        )
        if "test_metrics" in experiment:
            required.append(experiment["test_metrics"])

    for experiment in EFFICIENTNET_EXPERIMENTS:
        required.append(experiment["history"])
        if "validation_metrics" in experiment:
            required.append(experiment["validation_metrics"])

    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing result files:\n{missing_text}")


def build_experiment_table() -> list[dict[str, str | int | float]]:
    rows = []
    for experiment in EXPERIMENTS:
        history = read_csv(experiment["history"])
        validation = read_json(experiment["validation_metrics"])
        best_history_row = max(history, key=lambda row: float(row["val_accuracy"]))

        row: dict[str, str | int | float] = {
            "experiment": experiment["name"],
            "trainable_scope": experiment["scope"],
            "learning_rate": experiment["learning_rate"],
            "epochs": len(history),
            "best_epoch": int(best_history_row["epoch"]),
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_weighted_f1": validation["weighted_f1"],
            "val_top5_accuracy": validation["top5_accuracy"],
            "test_accuracy": "",
            "test_macro_f1": "",
            "test_weighted_f1": "",
            "test_top5_accuracy": "",
        }

        if "test_metrics" in experiment:
            test = read_json(experiment["test_metrics"])
            row.update(
                {
                    "test_accuracy": test["accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "test_weighted_f1": test["weighted_f1"],
                    "test_top5_accuracy": test["top5_accuracy"],
                }
            )
        rows.append(row)

    write_table(OUTPUT_DIR / "experiments.csv", rows)
    write_table(TABLE_DIR / "resnet18_experiments.csv", rows)
    return rows


def write_table(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to a CSV file with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_efficientnet_table() -> list[dict[str, str | int | float]]:
    """Build validation-only EfficientNet-B0 experiment rows."""
    rows = []
    for experiment in EFFICIENTNET_EXPERIMENTS:
        history = read_csv(experiment["history"])
        best_history_row = max(history, key=lambda row: float(row["val_accuracy"]))
        validation = (
            read_json(experiment["validation_metrics"])
            if "validation_metrics" in experiment
            else None
        )
        rows.append(
            {
                "experiment": experiment["name"],
                "trainable_scope": experiment["scope"],
                "learning_rate": experiment["learning_rate"],
                "epochs": len(history),
                "best_epoch": int(best_history_row["epoch"]),
                "val_accuracy": (
                    validation["accuracy"]
                    if validation is not None
                    else float(best_history_row["val_accuracy"])
                ),
                "val_macro_f1": (
                    validation["macro_f1"] if validation is not None else ""
                ),
                "val_weighted_f1": (
                    validation["weighted_f1"] if validation is not None else ""
                ),
                "val_top5_accuracy": (
                    validation["top5_accuracy"] if validation is not None else ""
                ),
                "evaluation_scope": "validation_only",
                "test_evaluated": False,
            }
        )

    write_table(TABLE_DIR / "efficientnet_b0_experiments.csv", rows)
    return rows


def write_resnet_test_tables() -> None:
    """Copy the locked ResNet18 test metrics into tracked report tables."""
    test_metrics = read_json(EXPERIMENTS[-1]["test_metrics"])
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
        "checkpoint",
        "config",
    ]
    write_table(
        TABLE_DIR / "resnet18_final_test_metrics.csv",
        [{field: test_metrics[field] for field in metric_fields}],
    )
    write_table(
        TABLE_DIR / "resnet18_test_per_class_metrics.csv",
        read_csv(TEST_PER_CLASS_PATH),
    )


def plot_training_curves() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)

    for row_index, experiment in enumerate(EXPERIMENTS):
        history = read_csv(experiment["history"])
        epochs = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]
        train_accuracy = [float(row["train_accuracy"]) * 100 for row in history]
        val_accuracy = [float(row["val_accuracy"]) * 100 for row in history]

        loss_axis = axes[row_index, 0]
        accuracy_axis = axes[row_index, 1]

        loss_axis.plot(epochs, train_loss, marker="o", label="Train")
        loss_axis.plot(epochs, val_loss, marker="s", label="Validation")
        loss_axis.set_title(f"{experiment['name']} - Loss")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Cross-entropy loss")
        loss_axis.set_xticks(epochs)
        loss_axis.grid(alpha=0.25)
        loss_axis.legend()

        accuracy_axis.plot(epochs, train_accuracy, marker="o", label="Train")
        accuracy_axis.plot(epochs, val_accuracy, marker="s", label="Validation")
        accuracy_axis.set_title(f"{experiment['name']} - Accuracy")
        accuracy_axis.set_xlabel("Epoch")
        accuracy_axis.set_ylabel("Accuracy (%)")
        accuracy_axis.set_xticks(epochs)
        accuracy_axis.grid(alpha=0.25)
        accuracy_axis.legend()

    fig.suptitle("ResNet18 Training and Validation Curves", fontsize=16)
    fig.savefig(FIGURE_DIR / "resnet18_training_curves.png", dpi=300)
    plt.close(fig)


def plot_validation_comparison(rows: list[dict[str, str | int | float]]) -> None:
    names = [str(row["experiment"]) for row in rows]
    metric_specs = [
        ("val_accuracy", "Accuracy"),
        ("val_macro_f1", "Macro-F1"),
        ("val_weighted_f1", "Weighted-F1"),
    ]
    x_positions = list(range(len(names)))
    width = 0.24

    fig, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    for metric_index, (field, label) in enumerate(metric_specs):
        positions = [
            position + (metric_index - 1) * width for position in x_positions
        ]
        values = [float(row[field]) * 100 for row in rows]
        bars = axis.bar(positions, values, width=width, label=label)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    axis.set_title("ResNet18 Validation Performance")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(x_positions, names)
    axis.set_ylim(85, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper center")
    fig.savefig(FIGURE_DIR / "resnet18_validation_comparison.png", dpi=300)
    plt.close(fig)


def plot_test_per_class_f1() -> None:
    rows = read_csv(TEST_PER_CLASS_PATH)
    rows.sort(key=lambda row: float(row["f1"]))

    ranked_path = OUTPUT_DIR / "test_per_class_ranked.csv"
    with ranked_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    names = [row["class_name"].replace("_", " ") for row in rows]
    scores = [float(row["f1"]) * 100 for row in rows]
    colors = ["#d97706" if index < 5 else "#2563eb" for index in range(len(rows))]

    fig, axis = plt.subplots(figsize=(10, 11), constrained_layout=True)
    bars = axis.barh(names, scores, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=7)
    axis.set_title("Final ResNet18 Test F1 by Species")
    axis.set_xlabel("F1 score (%)")
    axis.set_xlim(0, 105)
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(FIGURE_DIR / "resnet18_test_per_class_f1.png", dpi=300)
    plt.close(fig)


def plot_efficientnet_training_curves() -> None:
    """Plot frozen and fine-tuned EfficientNet-B0 histories."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, experiment in enumerate(EFFICIENTNET_EXPERIMENTS):
        history = read_csv(experiment["history"])
        epochs = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]
        train_accuracy = [float(row["train_accuracy"]) * 100 for row in history]
        val_accuracy = [float(row["val_accuracy"]) * 100 for row in history]

        axes[row_index, 0].plot(epochs, train_loss, marker="o", label="Train")
        axes[row_index, 0].plot(epochs, val_loss, marker="s", label="Validation")
        axes[row_index, 0].set_title(f"{experiment['name']} - Loss")
        axes[row_index, 0].set_ylabel("Cross-entropy loss")

        axes[row_index, 1].plot(epochs, train_accuracy, marker="o", label="Train")
        axes[row_index, 1].plot(epochs, val_accuracy, marker="s", label="Validation")
        axes[row_index, 1].set_title(f"{experiment['name']} - Accuracy")
        axes[row_index, 1].set_ylabel("Accuracy (%)")

        for axis in axes[row_index]:
            axis.set_xlabel("Epoch")
            axis.set_xticks(epochs)
            axis.grid(alpha=0.25)
            axis.legend()

    fig.suptitle("EfficientNet-B0 Training and Validation Curves", fontsize=16)
    fig.savefig(
        EFFICIENTNET_FIGURE_DIR / "efficientnet_b0_training_curves.png",
        dpi=300,
    )
    plt.close(fig)


def plot_model_validation_comparison(
    resnet_rows: list[dict[str, str | int | float]],
    efficientnet_rows: list[dict[str, str | int | float]],
) -> None:
    """Compare locked ResNet18 and validation-only EfficientNet-B0 results."""
    resnet = resnet_rows[-1]
    efficientnet = efficientnet_rows[-1]
    names = ["ResNet18", "EfficientNet-B0\n(validation only)"]
    accuracies = [
        float(resnet["val_accuracy"]) * 100,
        float(efficientnet["val_accuracy"]) * 100,
    ]
    macro_f1 = [
        float(resnet["val_macro_f1"]) * 100,
        float(efficientnet["val_macro_f1"]) * 100,
    ]

    x_positions = [0, 1]
    width = 0.34
    fig, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    accuracy_bars = axis.bar(
        [position - width / 2 for position in x_positions],
        accuracies,
        width=width,
        label="Validation accuracy",
    )
    f1_bars = axis.bar(
        [position + width / 2 for position in x_positions],
        macro_f1,
        width=width,
        label="Validation Macro-F1",
    )
    axis.bar_label(accuracy_bars, fmt="%.2f", padding=3)
    axis.bar_label(f1_bars, fmt="%.2f", padding=3)
    axis.set_title("Transfer Model Validation Comparison")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(x_positions, names)
    axis.set_ylim(95, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(
        EFFICIENTNET_FIGURE_DIR / "transfer_model_validation_comparison.png",
        dpi=300,
    )
    plt.close(fig)


def main() -> None:
    require_files()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EFFICIENTNET_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_rows = build_experiment_table()
    efficientnet_rows = build_efficientnet_table()
    write_resnet_test_tables()
    plot_training_curves()
    plot_validation_comparison(experiment_rows)
    plot_test_per_class_f1()
    plot_efficientnet_training_curves()
    plot_model_validation_comparison(experiment_rows, efficientnet_rows)

    print(f"Experiment table: {OUTPUT_DIR / 'experiments.csv'}")
    print(f"Ranked per-class table: {OUTPUT_DIR / 'test_per_class_ranked.csv'}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"EfficientNet-B0 figures: {EFFICIENTNET_FIGURE_DIR}")
    print(f"Tracked tables: {TABLE_DIR}")
    print("No model training or dataset evaluation was performed.")


if __name__ == "__main__":
    main()
