import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import __version__ as pillow_version
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import LeafDataset, load_samples_from_csv
from src.evaluation.error_analysis import prediction_rows_from_logits, save_error_rows
from src.evaluation.metrics import classification_metrics_from_logits
from src.models.baseline_cnn import CustomCNN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path) -> dict[str, int]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    result = {str(name): int(index) for name, index in mapping.items()}
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError("Class mapping must use contiguous indices from zero.")
    return result


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows or any(not row.get("path") or not row.get("label") for row in rows):
        raise ValueError(f"Invalid or empty split file: {path}")
    return rows


def load_sources(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    result = {row["path"]: row["source"] for row in rows}
    invalid = sorted(set(result.values()) - {"lab", "field"})
    if invalid:
        raise ValueError(f"Invalid source values in metadata: {invalid}")
    return result


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    return checkpoint


def export_predictions(
    *,
    checkpoint_path: Path,
    data_root: Path,
    test_split_path: Path,
    class_mapping_path: Path,
    image_metadata_path: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    reported_summary_path: Path | None = None,
) -> dict[str, Any]:
    class_to_idx = load_mapping(class_mapping_path)
    class_names = [""] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        class_names[class_index] = class_name

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_mapping = {
        str(name): int(index)
        for name, index in checkpoint.get("class_to_idx", {}).items()
    }
    if checkpoint_mapping != class_to_idx:
        raise ValueError("Checkpoint and locked class mappings do not match.")

    settings = checkpoint.get("settings", {})
    image_size = int(settings.get("image_size", 128))
    split_rows = read_split_rows(test_split_path)
    samples = load_samples_from_csv(test_split_path, data_root=data_root)
    missing = [str(path) for path, _ in samples if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test images are missing; first missing path: {missing[0]}"
        )

    source_by_path = load_sources(image_metadata_path)
    try:
        sources = [source_by_path[row["path"]] for row in split_rows]
    except KeyError as error:
        raise ValueError(f"Test path is absent from image metadata: {error.args[0]}") from error

    model = CustomCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target_layer = model.features[3][3]
    if not isinstance(target_layer, nn.Conv2d):
        raise TypeError("Expected model.features[3][3] to be a Conv2d layer.")
    model.to(device).eval()

    dataset = LeafDataset(
        samples,
        class_to_idx,
        image_size=image_size,
        training=False,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_batches = []
    target_batches = []
    with torch.inference_mode():
        for images, targets in loader:
            logits_batches.append(model(images.to(device)).cpu())
            target_batches.append(targets)

    logits = torch.cat(logits_batches)
    targets = torch.cat(target_batches)
    relative_paths = [row["path"] for row in split_rows]
    prediction_rows = prediction_rows_from_logits(
        logits,
        targets,
        class_names,
        image_paths=relative_paths,
        sources=sources,
    )
    metrics, _, _ = classification_metrics_from_logits(logits, targets, class_names)
    test_loss = torch.nn.functional.cross_entropy(logits, targets).item()

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "per_image_predictions.csv"
    save_error_rows(prediction_path, prediction_rows)

    reported_result = None
    if reported_summary_path is not None:
        reported_summary = json.loads(
            reported_summary_path.read_text(encoding="utf-8")
        )
        reported_result = {
            "summary_file": str(reported_summary_path),
            "test_loss": float(reported_summary["test_loss"]),
            "test_accuracy": float(reported_summary["test_acc"]),
            "recomputed_loss_difference": (
                test_loss - float(reported_summary["test_loss"])
            ),
            "recomputed_accuracy_difference": (
                float(metrics["accuracy"]) - float(reported_summary["test_acc"])
            ),
        }

    manifest = {
        "model": "Custom CNN trained from scratch",
        "purpose": "Post-hoc inference on the locked test split; no model selection or tuning.",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_validation_accuracy": checkpoint.get("best_val_acc"),
        "class_mapping_file": str(class_mapping_path),
        "test_split_file": str(test_split_path),
        "image_metadata_file": str(image_metadata_path),
        "num_test_images": len(split_rows),
        "source_counts": {
            source: sources.count(source) for source in ("lab", "field")
        },
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "runtime": {
            "torch": torch.__version__,
            "pillow": pillow_version,
        },
        "gradcam_target_layer": "model.features[3][3]",
        "test_loss": test_loss,
        "metrics": metrics,
        "reported_result": reported_result,
        "prediction_file": str(prediction_path),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reproducible per-image predictions for the Baseline model."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reported-summary", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    manifest = export_predictions(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        test_split_path=args.test_split,
        class_mapping_path=args.class_mapping,
        image_metadata_path=args.image_metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        reported_summary_path=args.reported_summary,
    )
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"Test loss: {manifest['test_loss']:.12f}")
    print(f"Saved predictions to {manifest['prediction_file']}")


if __name__ == "__main__":
    main()
