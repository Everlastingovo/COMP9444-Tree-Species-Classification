import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from part5_benchmark_efficiency import (
    LockedModel,
    build_locked_model,
    locked_models,
)
from part5_export_baseline_predictions import load_mapping
from src.evaluation.metrics import (
    classification_metrics_from_logits,
    metrics_by_source,
)
from src.evaluation.preprocessing import preprocess_evaluation_image


class BrightnessDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        data_root: Path,
        class_to_idx: dict[str, int],
        model_config: LockedModel,
        brightness_factor: float,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.model_config = model_config
        self.brightness_factor = brightness_factor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = self.data_root / row["path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            tensor = preprocess_evaluation_image(
                image,
                image_size=self.model_config.image_size,
                normalization=self.model_config.normalization,
                brightness_factor=self.brightness_factor,
            )
        target = torch.tensor(
            self.class_to_idx[row["label"]],
            dtype=torch.long,
        )
        return tensor, target


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"CSV file cannot be empty: {path}")
    return rows


def load_sources(metadata_path: Path) -> dict[str, str]:
    rows = read_csv(metadata_path)
    sources = {row["path"]: row["source"] for row in rows}
    invalid = sorted(set(sources.values()) - {"lab", "field"})
    if invalid:
        raise ValueError(f"Invalid image sources: {invalid}")
    return sources


def evaluate_brightness(
    model: torch.nn.Module,
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    logits_batches = []
    target_batches = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            logits_batches.append(model(images))
            target_batches.append(targets)
    elapsed_seconds = time.perf_counter() - start
    return (
        torch.cat(logits_batches),
        torch.cat(target_batches),
        elapsed_seconds,
    )


def run_brightness_experiment(
    *,
    repo_root: Path,
    data_root: Path,
    test_split_path: Path,
    class_mapping_path: Path,
    image_metadata_path: Path,
    output_dir: Path,
    brightness_factors: list[float],
    batch_size: int,
) -> list[dict[str, Any]]:
    if not brightness_factors or any(factor <= 0 for factor in brightness_factors):
        raise ValueError("Brightness factors must contain positive values.")
    if 1.0 not in brightness_factors:
        raise ValueError("Brightness factors must include the clean 1.0 condition.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    class_to_idx = load_mapping(class_mapping_path)
    class_names = [""] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        class_names[class_index] = class_name

    split_rows = read_csv(test_split_path)
    source_by_path = load_sources(image_metadata_path)
    try:
        sources = [source_by_path[row["path"]] for row in split_rows]
    except KeyError as error:
        raise ValueError(f"Missing source metadata for {error.args[0]}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    for config in locked_models(repo_root):
        model = build_locked_model(config, class_to_idx)
        for factor in brightness_factors:
            print(
                f"Evaluating {config.name} at brightness {factor:.2f}...",
                flush=True,
            )
            dataset = BrightnessDataset(
                split_rows,
                data_root=data_root,
                class_to_idx=class_to_idx,
                model_config=config,
                brightness_factor=factor,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            logits, targets, elapsed_seconds = evaluate_brightness(model, loader)
            overall, _, _ = classification_metrics_from_logits(
                logits,
                targets,
                class_names,
            )
            by_source = metrics_by_source(
                logits,
                targets,
                sources,
                class_names,
            )
            result_rows.append(
                {
                    "model": config.name,
                    "brightness_factor": factor,
                    "num_samples": int(overall["num_samples"]),
                    "accuracy": float(overall["accuracy"]),
                    "macro_f1": float(overall["macro_f1"]),
                    "weighted_f1": float(overall["weighted_f1"]),
                    "top5_accuracy": float(overall["top5_accuracy"]),
                    "lab_accuracy": float(by_source["lab"]["accuracy"]),
                    "lab_macro_f1": float(by_source["lab"]["macro_f1"]),
                    "field_accuracy": float(by_source["field"]["accuracy"]),
                    "field_macro_f1": float(by_source["field"]["macro_f1"]),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            save_rows(output_dir / "brightness_robustness.csv", result_rows)
        del model

    add_clean_differences(result_rows)
    save_rows(output_dir / "brightness_robustness.csv", result_rows)
    summaries = summarize_robustness(result_rows)
    save_rows(output_dir / "brightness_robustness_summary.csv", summaries)
    plot_brightness_robustness(
        output_dir / "brightness_robustness.png",
        result_rows,
    )
    plot_source_robustness(
        output_dir / "brightness_source_robustness.png",
        result_rows,
    )
    manifest = {
        "purpose": (
            "Post-hoc brightness stress test on frozen, validation-selected "
            "checkpoints; no model selection or tuning."
        ),
        "device": "cpu",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "brightness_factors": brightness_factors,
        "clean_factor": 1.0,
        "batch_size": batch_size,
        "num_test_images": len(split_rows),
        "source_counts": {
            source: sources.count(source) for source in ("lab", "field")
        },
        "perturbation_order": (
            "Brightness is adjusted on the RGB image before locked resize, "
            "padding, tensor conversion, and normalization."
        ),
        "limitations": (
            "Synthetic global brightness does not reproduce all real-world "
            "illumination changes such as shadows, glare, or color shifts."
        ),
    }
    (output_dir / "brightness_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return result_rows


def add_clean_differences(rows: list[dict[str, Any]]) -> None:
    clean_accuracy = {
        str(row["model"]): float(row["accuracy"])
        for row in rows
        if float(row["brightness_factor"]) == 1.0
    }
    for row in rows:
        row["accuracy_change_from_clean"] = (
            float(row["accuracy"]) - clean_accuracy[str(row["model"])]
        )


def summarize_robustness(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for model_name in dict.fromkeys(str(row["model"]) for row in rows):
        model_rows = [row for row in rows if row["model"] == model_name]
        clean = next(
            row
            for row in model_rows
            if float(row["brightness_factor"]) == 1.0
        )
        perturbed = [
            row
            for row in model_rows
            if float(row["brightness_factor"]) != 1.0
        ]
        worst = min(model_rows, key=lambda row: float(row["accuracy"]))
        summaries.append(
            {
                "model": model_name,
                "clean_accuracy": float(clean["accuracy"]),
                "mean_perturbed_accuracy": sum(
                    float(row["accuracy"]) for row in perturbed
                )
                / len(perturbed),
                "worst_brightness_factor": float(
                    worst["brightness_factor"]
                ),
                "worst_accuracy": float(worst["accuracy"]),
                "maximum_accuracy_drop": (
                    float(clean["accuracy"]) - float(worst["accuracy"])
                ),
            }
        )
    return summaries


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_brightness_robustness(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Baseline-CustomCNN": "#2A6F97",
        "MobileNetV2-Partial": "#40916C",
        "ResNet18-Layer4": "#D97706",
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model_name in dict.fromkeys(str(row["model"]) for row in rows):
        model_rows = sorted(
            (row for row in rows if row["model"] == model_name),
            key=lambda row: float(row["brightness_factor"]),
        )
        axis.plot(
            [float(row["brightness_factor"]) for row in model_rows],
            [float(row["accuracy"]) for row in model_rows],
            marker="o",
            linewidth=2,
            label=model_name,
            color=colors[model_name],
        )
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_title("Brightness Robustness on the Locked Test Split")
    axis.set_xlabel("Brightness factor (1.0 = original)")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(
        sorted({float(row["brightness_factor"]) for row in rows})
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_source_robustness(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(dict.fromkeys(str(row["model"]) for row in rows))
    figure, axes = plt.subplots(1, len(model_names), figsize=(15, 4.8), sharey=True)
    for axis, model_name in zip(axes, model_names):
        model_rows = sorted(
            (row for row in rows if row["model"] == model_name),
            key=lambda row: float(row["brightness_factor"]),
        )
        factors = [float(row["brightness_factor"]) for row in model_rows]
        axis.plot(
            factors,
            [float(row["lab_accuracy"]) for row in model_rows],
            marker="o",
            linewidth=2,
            label="Lab",
            color="#2A6F97",
        )
        axis.plot(
            factors,
            [float(row["field_accuracy"]) for row in model_rows],
            marker="s",
            linewidth=2,
            label="Field",
            color="#D97706",
        )
        axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(model_name)
        axis.set_xlabel("Brightness factor")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend()
    figure.suptitle("Lab and Field Brightness Robustness")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_factors(value: str) -> list[float]:
    factors = [float(item.strip()) for item in value.split(",") if item.strip()]
    return sorted(set(factors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen-checkpoint brightness robustness stress test."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--test-split",
        type=Path,
        default=Path("data/splits/test.csv"),
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/metadata/class_to_idx.json"),
    )
    parser.add_argument(
        "--image-metadata",
        type=Path,
        default=Path("data/metadata/images.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison/experiments"),
    )
    parser.add_argument(
        "--brightness-factors",
        type=parse_factors,
        default=parse_factors("0.6,0.8,1.0,1.2,1.4"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_brightness_experiment(
        repo_root=args.repo_root.resolve(),
        data_root=args.data_root,
        test_split_path=args.test_split,
        class_mapping_path=args.class_mapping,
        image_metadata_path=args.image_metadata,
        output_dir=args.output_dir,
        brightness_factors=args.brightness_factors,
        batch_size=args.batch_size,
    )
    print(json.dumps(summarize_robustness(rows), indent=2))
    print(f"Saved brightness outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
