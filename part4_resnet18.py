import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import LeafDataset, load_samples_from_csv
from src.models.resnet18 import build_resnet18, count_parameters
from src.training.trainer import run_epoch
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_device
from src.utils.seed import set_seed


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment settings from a YAML file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_class_mapping(mapping_path: Path) -> dict[str, int]:
    """Load the fixed 30-class mapping created by Part 2."""
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)


def make_loader(
    samples,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    training: bool,
    num_workers: int,
    normalization: str,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader without modifying the fixed dataset splits."""
    dataset = LeafDataset(
        samples=samples,
        class_to_idx=class_to_idx,
        image_size=image_size,
        training=training,
        augment=training,
        normalization=normalization,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def write_history(history: list[dict[str, float]], output_path: Path) -> None:
    """Save epoch-level training and validation metrics."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on the fixed Leafsnap data splits."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resnet18_frozen.yaml"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs in the config file.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for a smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory in the config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    epochs = (
        args.epochs
        if args.epochs is not None
        else training_config["epochs"]
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(output_config["dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = training_config["seed"]
    set_seed(seed)

    device = get_device()

    split_dir = Path("data/splits")
    mapping_path = Path("data/metadata/class_to_idx.json")

    train_samples = load_samples_from_csv(split_dir / "train.csv")
    val_samples = load_samples_from_csv(split_dir / "val.csv")
    test_samples = load_samples_from_csv(split_dir / "test.csv")
    class_to_idx = load_class_mapping(mapping_path)

    if len(class_to_idx) != model_config["num_classes"]:
        raise ValueError(
            f"Expected {model_config['num_classes']} classes, "
            f"but class_to_idx contains {len(class_to_idx)} classes."
        )

    print(f"Device: {device}")
    print(f"Classes: {len(class_to_idx)}")
    print(f"Train images: {len(train_samples)}")
    print(f"Validation images: {len(val_samples)}")
    print(f"Test images: {len(test_samples)}")
    print(f"Normalization: {model_config['normalization']}")
    print(f"Trainable scope: {model_config['trainable_scope']}")

    train_loader = make_loader(
        samples=train_samples,
        class_to_idx=class_to_idx,
        image_size=model_config["image_size"],
        batch_size=training_config["batch_size"],
        training=True,
        num_workers=training_config["num_workers"],
        normalization=model_config["normalization"],
        device=device,
    )

    val_loader = make_loader(
        samples=val_samples,
        class_to_idx=class_to_idx,
        image_size=model_config["image_size"],
        batch_size=training_config["batch_size"],
        training=False,
        num_workers=training_config["num_workers"],
        normalization=model_config["normalization"],
        device=device,
    )

    model = build_resnet18(
        num_classes=model_config["num_classes"],
        trainable_scope=model_config["trainable_scope"],
        pretrained=model_config["pretrained"],
    ).to(device)

    initial_checkpoint = model_config.get("initial_checkpoint")
    if initial_checkpoint:
        initial_checkpoint_path = Path(initial_checkpoint)
        if not initial_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint_path}"
            )

        checkpoint = load_checkpoint(initial_checkpoint_path, device=device)
        if checkpoint["class_to_idx"] != class_to_idx:
            raise ValueError(
                "Checkpoint class mapping does not match the current class mapping."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded initial checkpoint: {initial_checkpoint_path}")

    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    criterion = nn.CrossEntropyLoss()

    parameters_to_optimize = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        parameters_to_optimize,
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_batches,
        )

        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_batches,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_result)

        print(
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint_settings = {
                "experiment_name": config["experiment_name"],
                "model": model_config,
                "training": training_config,
                "normalization": model_config["normalization"],
                "trainable_scope": model_config["trainable_scope"],
                "seed": seed,
            }

            save_checkpoint(
                best_model_path,
                model=model,
                class_to_idx=class_to_idx,
                settings=checkpoint_settings,
                best_val_acc=best_val_accuracy,
            )

            print(f"Saved best checkpoint: {best_model_path}")

    write_history(history, output_dir / "history.csv")

    summary = {
        "experiment_name": config["experiment_name"],
        "best_val_accuracy": best_val_accuracy,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "initial_checkpoint": initial_checkpoint,
        "epochs_completed": epochs,
        "max_batches": args.max_batches,
        "test_set_evaluated": False,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nTraining completed")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"History saved to: {output_dir / 'history.csv'}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
