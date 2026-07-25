import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from part5_export_baseline_predictions import load_checkpoint, load_mapping
from src.data.transforms import LeafImageTransform
from src.evaluation.gradcam import GradCAM, save_gradcam_grid
from src.models.baseline_cnn import CustomCNN


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("Prediction file cannot be empty.")
    return rows


def row_is_correct(row: dict[str, str]) -> bool:
    return row["correct"].strip().lower() in {"true", "1", "yes"}


def select_examples(
    rows: list[dict[str, str]],
    examples_per_group: int,
) -> list[dict[str, str]]:
    selected = []
    for source in ("lab", "field"):
        for correct in (True, False):
            group = [
                row
                for row in rows
                if row["source"] == source and row_is_correct(row) is correct
            ]
            group.sort(key=lambda row: (-float(row["confidence"]), row["path"]))
            selected.extend(group[:examples_per_group])
    if not selected:
        raise ValueError("No Grad-CAM examples matched the selection groups.")
    return selected


def save_selection(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_baseline_gradcam(
    *,
    checkpoint_path: Path,
    predictions_path: Path,
    class_mapping_path: Path,
    data_root: Path,
    output_dir: Path,
    examples_per_group: int,
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

    model = CustomCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    target_layer = model.features[3][3]

    selected = select_examples(
        read_predictions(predictions_path),
        examples_per_group=examples_per_group,
    )
    image_size = int(checkpoint.get("settings", {}).get("image_size", 128))
    transform = LeafImageTransform(
        image_size=image_size,
        training=False,
        augment=False,
    )

    original_images = []
    input_tensors = []
    for row in selected:
        image_path = data_root / row["path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Grad-CAM image not found: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            original_images.append(image.copy())
            input_tensors.append(transform(image))

    images = torch.stack(input_tensors)
    explained_classes = [int(row["predicted_idx"]) for row in selected]
    with GradCAM(model, target_layer=target_layer) as gradcam:
        heatmaps, logits, _ = gradcam.generate(
            images,
            class_indices=explained_classes,
        )

    recomputed_predictions = logits.argmax(dim=1).tolist()
    if recomputed_predictions != explained_classes:
        raise RuntimeError(
            "Current model predictions do not match the saved prediction rows."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_gradcam_grid(
        output_dir / "gradcam_examples.png",
        original_images,
        heatmaps,
        [row["true_label"] for row in selected],
        [row["predicted_label"] for row in selected],
        [float(row["confidence"]) for row in selected],
    )

    selection_rows = []
    for row in selected:
        selection_rows.append(
            {
                "path": row["path"],
                "source": row["source"],
                "correct": row["correct"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "explained_class": row["predicted_label"],
                "target_layer": "model.features[3][3]",
            }
        )
    save_selection(output_dir / "gradcam_selection.csv", selection_rows)

    summary = {
        "model": "Baseline-CustomCNN",
        "checkpoint_file": checkpoint_path.name,
        "predictions_file": str(predictions_path),
        "target_layer": "model.features[3][3]",
        "explained_class": "predicted class",
        "selection_rule": (
            "Highest-confidence correct and incorrect predictions per image source."
        ),
        "num_examples": len(selected),
        "examples_per_group": examples_per_group,
    }
    (output_dir / "gradcam_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible Baseline Grad-CAM examples."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/metadata/class_to_idx.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-group", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples_per_group <= 0:
        raise ValueError("examples-per-group must be positive.")
    summary = generate_baseline_gradcam(
        checkpoint_path=args.checkpoint,
        predictions_path=args.predictions,
        class_mapping_path=args.class_mapping,
        data_root=args.data_root,
        output_dir=args.output_dir,
        examples_per_group=args.examples_per_group,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved Grad-CAM outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
