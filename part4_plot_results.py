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

FIGURE_DIR = Path("report/figures/resnet18")
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

    path = OUTPUT_DIR / "experiments.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


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
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
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


def main() -> None:
    require_files()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_rows = build_experiment_table()
    plot_training_curves()
    plot_validation_comparison(experiment_rows)
    plot_test_per_class_f1()

    print(f"Experiment table: {OUTPUT_DIR / 'experiments.csv'}")
    print(f"Ranked per-class table: {OUTPUT_DIR / 'test_per_class_ranked.csv'}")
    print(f"Figures: {FIGURE_DIR}")
    print("No model training or dataset evaluation was performed.")


if __name__ == "__main__":
    main()
