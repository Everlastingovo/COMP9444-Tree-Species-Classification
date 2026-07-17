import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LeafDataset, load_samples_from_csv
from src.models.resnet18 import build_resnet18
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ResNet18 checkpoint on validation or test data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resnet18_layer4_lr3e5.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/resnet18/layer4_lr3e5/best_model.pt"),
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
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
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
            batch_probabilities = torch.softmax(logits, dim=1)

            targets.append(batch_targets.cpu())
            probabilities.append(batch_probabilities.cpu())

            if max_batches is not None and batch_index >= max_batches:
                break

    if not targets:
        raise RuntimeError(f"{split.title()} loader produced no batches.")

    return (
        torch.cat(targets).numpy(),
        torch.cat(probabilities).numpy(),
    )


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float | int], tuple[np.ndarray, ...]]:
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

    top5_predictions = np.argpartition(probabilities, -5, axis=1)[:, -5:]
    top5_accuracy = np.mean(
        [target in candidates for target, candidates in zip(targets, top5_predictions)]
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
    return metrics, per_class


def write_per_class_metrics(
    path: Path,
    per_class: tuple[np.ndarray, ...],
    class_to_idx: dict[str, int],
) -> None:
    precision, recall, f1, support = per_class
    idx_to_class = {class_idx: name for name, class_idx in class_to_idx.items()}

    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["class_idx", "class_name", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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


def write_predictions(
    path: Path,
    samples,
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_to_idx: dict[str, int],
) -> None:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    idx_to_class = {class_idx: name for name, class_idx in class_to_idx.items()}

    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "path",
            "true_class",
            "true_idx",
            "predicted_class",
            "predicted_idx",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
    with mapping_path.open("r", encoding="utf-8") as file:
        expected_mapping = json.load(file)
    if class_to_idx != expected_mapping:
        raise ValueError("Checkpoint class mapping does not match Part 2 metadata.")

    split_filename = "val.csv" if args.split == "validation" else "test.csv"
    evaluation_samples = load_samples_from_csv(Path("data/splits") / split_filename)
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

    model = build_resnet18(
        num_classes=model_config["num_classes"],
        trainable_scope=model_config["trainable_scope"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Evaluation split: {args.split}")
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
    metrics, per_class = calculate_metrics(
        targets,
        probabilities,
        model_config["num_classes"],
    )

    metrics.update(
        {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "max_batches": args.max_batches,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    write_per_class_metrics(
        output_dir / "per_class_metrics.csv",
        per_class,
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
    if args.split == "validation":
        print("Test set was not evaluated.")
    else:
        print("Final test evaluation completed.")


if __name__ == "__main__":
    main()
