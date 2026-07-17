import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.build_metadata import extract_dataset, write_class_mapping, write_image_metadata
from src.data.dataset import LeafDataset, collect_image_paths, load_samples_from_csv
from src.data.split_dataset import stratified_split, write_dataset_split_csv, write_split_csvs
from src.models.baseline_cnn import CustomCNN
from src.training.losses import build_classification_loss
from src.training.scheduler import build_cosine_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_device
from src.utils.seed import set_seed


ModelFactory = Callable[[int, dict[str, Any]], nn.Module]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    progress = tqdm(loader, leave=False)
    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size
        progress.set_description(f"loss={total_loss / total_seen:.4f}, acc={total_correct / total_seen:.4f}")

        if max_batches is not None and batch_index >= max_batches:
            break

    return total_loss / total_seen, total_correct / total_seen


def main(
    model: nn.Module | None = None,
    model_factory: ModelFactory | None = None,
    default_config: Path = Path("configs/baseline.yaml"),
) -> None:
    if model is not None and model_factory is not None:
        raise ValueError("Pass either model or model_factory, not both.")

    args = parse_args(default_config)
    settings = build_settings(args)

    set_seed(settings["seed"])
    settings["output_dir"].mkdir(parents=True, exist_ok=True)

    device = get_device()
    if settings["images_root"] is not None:
        images_root = settings["images_root"]
    elif settings["data_root"] is not None:
        images_root = settings["data_root"] / "dataset" / "images"
    else:
        images_root = extract_dataset(settings["zip_path"], settings["extract_root"])
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    if settings["use_splits"]:
        split_dir = settings["split_dir"]
        train_samples = load_samples_from_csv(split_dir / "train.csv", settings["data_root"])
        val_samples = load_samples_from_csv(split_dir / "val.csv", settings["data_root"])
        test_samples = load_samples_from_csv(split_dir / "test.csv", settings["data_root"])
        samples = [*train_samples, *val_samples, *test_samples]
        classes = sorted({label for _, label in samples})
        class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
        splits = {"train": train_samples, "val": val_samples, "test": test_samples}
    else:
        samples = collect_image_paths(images_root, settings["image_source"])
        classes = sorted({label for _, label in samples})
        class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
        train_samples, val_samples, test_samples = stratified_split(
            samples,
            settings["val_ratio"],
            settings["test_ratio"],
            settings["seed"],
        )
        splits = {"train": train_samples, "val": val_samples, "test": test_samples}
        # data/splits and data/metadata are the locked, shared data foundation
        # (owned by the data-prep step) — only regenerate them here when this
        # run is doing its own ad-hoc split, never when reusing --use-splits.
        write_split_csvs(Path("data/splits"), splits)
        write_class_mapping(Path("data/metadata/class_to_idx.json"), class_to_idx)
        write_image_metadata(Path("data/metadata/images.csv"), samples, class_to_idx)

    image_source_label = "all (from data/splits)" if settings["use_splits"] else settings["image_source"]
    print(f"Device: {device}")
    print(f"Image source: {image_source_label}")
    print(f"Classes: {len(classes)}")
    print(f"Images: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    write_dataset_split_csv(settings["output_dir"] / "dataset_split.csv", splits)
    write_class_mapping(settings["output_dir"] / "class_to_idx.json", class_to_idx)

    train_loader = make_loader(train_samples, class_to_idx, settings, training=True, device=device)
    val_loader = make_loader(val_samples, class_to_idx, settings, training=False, device=device)
    test_loader = make_loader(test_samples, class_to_idx, settings, training=False, device=device)

    if model is None:
        model = (
            model_factory(len(classes), settings)
            if model_factory is not None
            else CustomCNN(num_classes=len(classes))
        )
    model = model.to(device)
    criterion = build_classification_loss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The supplied model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = build_cosine_scheduler(optimizer, settings["epochs"])

    best_val_acc = -1.0
    history = []
    start_time = time.time()
    best_model_path = settings["output_dir"] / f"best_{settings['model_name']}.pt"

    for epoch in range(1, settings["epochs"] + 1):
        print(f"\nEpoch {epoch}/{settings['epochs']}")
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=settings["max_batches"],
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=settings["max_batches"],
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_model_path,
                model=model,
                class_to_idx=class_to_idx,
                settings=settings,
                best_val_acc=best_val_acc,
            )
            print(f"Saved best model to {best_model_path}")

    checkpoint = load_checkpoint(best_model_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        max_batches=settings["max_batches"],
    )

    metrics_path = settings["output_dir"] / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": (
            "Custom CNN trained from scratch"
            if settings["model_name"] == "custom_cnn"
            else settings["model_name"]
        ),
        "image_source": settings["image_source"],
        "num_classes": len(classes),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "test_images": len(test_samples),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_minutes": (time.time() - start_time) / 60,
    }
    summary_path = settings["output_dir"] / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal result")
    print(json.dumps(summary, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Summary saved to {summary_path}")


def make_loader(
    samples: list[tuple[Path, str]],
    class_to_idx: dict[str, int],
    settings: dict[str, Any],
    training: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        LeafDataset(
            samples,
            class_to_idx,
            settings["image_size"],
            training=training,
            augment=not settings["no_augmentation"],
            normalization=settings["normalization"],
        ),
        batch_size=settings["batch_size"],
        shuffle=training,
        num_workers=settings["num_workers"],
        pin_memory=device.type == "cuda",
    )


def parse_args(default_config: Path = Path("configs/baseline.yaml")) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classification model on the locked Leafsnap splits.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--extract-root", type=Path, default=None)
    parser.add_argument("--image-source", choices=["field", "lab", "all"], default=None)
    parser.add_argument(
        "--use-splits",
        action="store_true",
        default=None,
        help="Load train/val/test samples from data/splits CSV files.",
    )
    parser.add_argument("--split-dir", type=Path, default=None, help="Directory containing train.csv, val.csv, and test.csv")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training-time augmentation for a deterministic baseline.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)

    settings = {
        "zip_path": Path(pick(args.zip_path, nested(config, "dataset", "zip_path"), "data/raw/5061353/leafsnap-dataset-30subset.zip")),
        "extract_root": Path(pick(args.extract_root, nested(config, "dataset", "extract_root"), "data/raw/5061353")),
        "data_root": optional_path(pick(args.data_root, nested(config, "dataset", "data_root"), None)),
        "images_root": optional_path(pick(args.images_root, nested(config, "dataset", "images_root"), None)),
        "image_source": pick(args.image_source, nested(config, "dataset", "image_source"), "field"),
        "use_splits": bool(pick(args.use_splits, nested(config, "dataset", "use_splits"), False)),
        "split_dir": Path(pick(args.split_dir, nested(config, "dataset", "split_dir"), "data/splits")),
        "no_augmentation": args.no_augmentation,
        "output_dir": Path(pick(args.output_dir, nested(config, "output", "dir"), "outputs/baseline")),
        "model_name": str(pick(None, nested(config, "model", "name"), "custom_cnn")),
        "pretrained": bool(pick(None, nested(config, "model", "pretrained"), False)),
        "fine_tune_mode": str(pick(None, nested(config, "model", "fine_tune_mode"), "full")),
        "unfreeze_blocks": int(pick(None, nested(config, "model", "unfreeze_blocks"), 4)),
        "image_size": int(pick(args.image_size, nested(config, "model", "image_size"), 128)),
        "normalization": str(pick(None, nested(config, "dataset", "normalization"), "baseline")),
        "epochs": int(pick(args.epochs, nested(config, "training", "epochs"), 20)),
        "batch_size": int(pick(args.batch_size, nested(config, "training", "batch_size"), 32)),
        "learning_rate": float(pick(args.learning_rate, nested(config, "training", "learning_rate"), 1e-3)),
        "weight_decay": float(pick(args.weight_decay, nested(config, "training", "weight_decay"), 1e-4)),
        "val_ratio": float(pick(args.val_ratio, nested(config, "dataset", "val_ratio"), 0.15)),
        "test_ratio": float(pick(args.test_ratio, nested(config, "dataset", "test_ratio"), 0.15)),
        "seed": int(pick(args.seed, nested(config, "training", "seed"), 42)),
        "num_workers": int(pick(args.num_workers, nested(config, "training", "num_workers"), 0)),
        "max_batches": pick(args.max_batches, nested(config, "training", "max_batches"), None),
    }
    return settings


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def pick(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)
