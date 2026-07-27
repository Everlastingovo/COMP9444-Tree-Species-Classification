import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LeafDataset, load_samples_from_csv
from src.models.efficientnet_b0 import build_efficientnet_b0
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an EfficientNet-B0 checkpoint on validation or test data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/efficientnet_b0_finetune.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/efficientnet_b0/finetune/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <configured output directory>/<split>.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluation split. Defaults to validation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit evaluation batches for a smoke test.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return evaluation targets and class probabilities."""
    model.eval()
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch_index, (images, batch_targets) in enumerate(
            tqdm(loader, desc=split.title(), leave=False),
            start=1,
        ):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            targets.append(batch_targets.cpu())
            probabilities.append(torch.softmax(logits, dim=1).cpu())

            if max_batches is not None and batch_index >= max_batches:
                break

    if not targets:
        raise RuntimeError(f"{split.title()} loader produced no batches.")

    return torch.cat(targets).numpy(), torch.cat(probabilities).numpy()


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float | int], tuple[np.ndarray, ...], np.ndarray]:
    """Calculate aggregate, per-class, and confusion-matrix metrics."""
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(num_classes)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )
    per_class = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )

    top_k = min(5, num_classes)
    top_predictions = np.argpartition(
        probabilities,
        -top_k,
        axis=1,
    )[:, -top_k:]
    top5_accuracy = np.mean(
        [target in candidates for target, candidates in zip(targets, top_predictions)]
    )

    metrics: dict[str, float | int] = {
        "num_samples": int(targets.size),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "top5_accuracy": float(top5_accuracy),
    }
    matrix = confusion_matrix(targets, predictions, labels=labels)
    return metrics, per_class, matrix


def write_per_class_metrics(
    path: Path,
    per_class: tuple[np.ndarray, ...],
    class_to_idx: dict[str, int],
) -> None:
    """Write precision, recall, F1, and support for every class."""
    precision, recall, f1, support = per_class
    idx_to_class = {class_idx: name for name, class_idx in class_to_idx.items()}

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["class_idx", "class_name", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for class_idx in range(len(class_to_idx)):
            writer.writerow(
                {
                    "class_idx": class_idx,
                    "class_name": idx_to_class[class_idx],
                    "precision": float(precision[class_idx]),
                    "recall": float(recall[class_idx]),
                    "f1": float(f1[class_idx]),
                    "support": int(support[class_idx]),
                }
            )


def write_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    class_to_idx: dict[str, int],
) -> None:
    """Write a labelled confusion matrix to CSV."""
    idx_to_class = {class_idx: name for name, class_idx in class_to_idx.items()}
    class_names = [idx_to_class[index] for index in range(len(class_to_idx))]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_class/predicted_class", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def write_predictions(
    path: Path,
    samples,
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_to_idx: dict[str, int],
) -> None:
    """Write one prediction row per evaluated image."""
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    idx_to_class = {class_idx: name for name, class_idx in class_to_idx.items()}

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "path",
            "true_class",
            "true_idx",
            "predicted_class",
            "predicted_idx",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (image_path, _), target, prediction, confidence in zip(
            samples,
            targets,
            predictions,
            confidences,
        ):
            writer.writerow(
                {
                    "path": image_path,
                    "true_class": idx_to_class[int(target)],
                    "true_idx": int(target),
                    "predicted_class": idx_to_class[int(prediction)],
                    "predicted_idx": int(prediction),
                    "confidence": float(confidence),
                    "correct": bool(target == prediction),
                }
            )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_config = config["model"]
    training_config = config["training"]
    output_dir = args.output_dir or Path(config["output"]["dir"]) / args.split

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    set_seed(training_config["seed"])
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    class_to_idx = checkpoint["class_to_idx"]

    mapping_path = Path("data/metadata/class_to_idx.json")
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        expected_mapping = json.load(mapping_file)
    if class_to_idx != expected_mapping:
        raise ValueError("Checkpoint class mapping does not match Part 2 metadata.")

    split_path = (
        Path("data/splits/val.csv")
        if args.split == "validation"
        else Path("data/splits/test.csv")
    )
    evaluation_samples = load_samples_from_csv(split_path)
    evaluation_dataset = LeafDataset(
        samples=evaluation_samples,
        class_to_idx=class_to_idx,
        image_size=model_config["image_size"],
        training=False,
        augment=False,
        normalization=model_config["normalization"],
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_efficientnet_b0(
        num_classes=model_config["num_classes"],
        trainable_scope=model_config["trainable_scope"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Evaluation split: {args.split}")
    print("Protocol: validation-based model selection followed by final test evaluation")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation images: {len(evaluation_samples)}")
    print(f"Normalization: {model_config['normalization']}")

    targets, probabilities = evaluate(
        model,
        evaluation_loader,
        device,
        args.max_batches,
        args.split,
    )
    metrics, per_class, matrix = calculate_metrics(
        targets,
        probabilities,
        model_config["num_classes"],
    )
    metrics.update(
        {
            "split": args.split,
            "evaluation_scope": (
                "validation_model_selection"
                if args.split == "validation"
                else "final_test"
            ),
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "max_batches": args.max_batches,
            "test_set_evaluated": args.split == "test",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    write_per_class_metrics(
        output_dir / "per_class_metrics.csv",
        per_class,
        class_to_idx,
    )
    write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        matrix,
        class_to_idx,
    )
    write_predictions(
        output_dir / "predictions.csv",
        evaluation_samples[: targets.size],
        targets,
        probabilities,
        class_to_idx,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Results saved to: {output_dir}")
    if args.split == "test":
        print("Final test evaluation completed.")
    else:
        print("Test set was not accessed or evaluated.")


if __name__ == "__main__":
    main()
