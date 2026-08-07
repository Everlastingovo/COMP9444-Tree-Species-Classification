import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v2, resnet18

from part5_export_baseline_predictions import load_checkpoint, load_mapping
from part5_generate_baseline_gradcam import save_selection, select_examples
from src.evaluation.gradcam import GradCAM, save_gradcam_grid
from src.evaluation.preprocessing import preprocess_evaluation_image


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("Prediction file cannot be empty.")
    return rows


def build_model(
    architecture: str,
    num_classes: int,
) -> tuple[nn.Module, nn.Module, str]:
    if architecture == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
        return model, model.features[-1], "model.features[-1]"

    if architecture == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, model.layer4[-1], "model.layer4[-1]"

    raise ValueError(f"Unsupported architecture: {architecture}")


def generate_transfer_gradcam(
    *,
    architecture: str,
    checkpoint_path: Path,
    predictions_path: Path,
    class_mapping_path: Path,
    data_root: Path,
    output_dir: Path,
    examples_per_group: int,
) -> dict[str, Any]:
    class_to_idx = load_mapping(class_mapping_path)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_mapping = {
        str(name): int(index)
        for name, index in checkpoint.get("class_to_idx", {}).items()
    }
    if checkpoint_mapping != class_to_idx:
        raise ValueError("Checkpoint and locked class mappings do not match.")

    model, target_layer, target_layer_name = build_model(
        architecture,
        len(class_to_idx),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    selected = select_examples(
        read_predictions(predictions_path),
        examples_per_group=examples_per_group,
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
            input_tensors.append(
                preprocess_evaluation_image(
                    image,
                    image_size=224,
                    normalization="imagenet",
                )
            )

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
                "target_layer": target_layer_name,
            }
        )
    save_selection(output_dir / "gradcam_selection.csv", selection_rows)

    summary = {
        "architecture": architecture,
        "checkpoint_file": checkpoint_path.name,
        "predictions_file": str(predictions_path),
        "target_layer": target_layer_name,
        "explained_class": "predicted class",
        "selection_rule": (
            "Highest-confidence correct and incorrect predictions per image source."
        ),
        "num_examples": len(selected),
        "examples_per_group": examples_per_group,
        "strict_checkpoint_load": True,
    }
    (output_dir / "gradcam_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM examples for MobileNetV2 or ResNet18."
    )
    parser.add_argument(
        "--architecture",
        choices=("mobilenetv2", "resnet18"),
        required=True,
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
    summary = generate_transfer_gradcam(
        architecture=args.architecture,
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
