import argparse
import csv
import json
import time
from dataclasses import dataclass
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
from src.utils.checkpoint import load_checkpoint, save_checkpoint, serializable_settings
from src.utils.device import get_device
from src.utils.seed import set_seed


ModelFactory = Callable[[int, dict[str, Any]], nn.Module]


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    top1: float
    top5: float
    macro_f1: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    num_classes: int = 30,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    progress = tqdm(loader, leave=False)
    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = targets.size(0)
        predictions = logits.argmax(dim=1)
        top5_predictions = logits.topk(k=min(5, num_classes), dim=1).indices
        total_loss += loss.item() * batch_size
        total_top1_correct += (predictions == targets).sum().item()
        total_top5_correct += (top5_predictions == targets.unsqueeze(1)).any(dim=1).sum().item()
        total_seen += batch_size
        encoded_pairs = targets.detach().cpu() * num_classes + predictions.detach().cpu()
        confusion += torch.bincount(encoded_pairs, minlength=num_classes**2).reshape(
            num_classes,
            num_classes,
        )
        progress.set_description(
            f"loss={total_loss / total_seen:.4f}, top1={total_top1_correct / total_seen:.4f}"
        )

        if max_batches is not None and batch_index >= max_batches:
            break

    if total_seen == 0:
        raise RuntimeError("The DataLoader produced no samples.")

    return EpochMetrics(
        loss=total_loss / total_seen,
        top1=total_top1_correct / total_seen,
        top5=total_top5_correct / total_seen,
        macro_f1=macro_f1_from_confusion(confusion),
    )


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    confusion = confusion.to(dtype=torch.float64)
    true_positives = confusion.diag()
    false_positives = confusion.sum(dim=0) - true_positives
    false_negatives = confusion.sum(dim=1) - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    per_class_f1 = torch.where(
        denominator > 0,
        2 * true_positives / denominator,
        torch.zeros_like(denominator),
    )
    return per_class_f1.mean().item()


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
        test_samples = None
        samples = [*train_samples, *val_samples]
        splits = {"train": train_samples, "val": val_samples}
        if settings["evaluate_test"]:
            test_samples = load_samples_from_csv(split_dir / "test.csv", settings["data_root"])
            samples.extend(test_samples)
            splits["test"] = test_samples
        classes = sorted({label for _, label in samples})
        class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
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
    split_message = f"Images: train={len(train_samples)}, val={len(val_samples)}"
    if test_samples is not None:
        split_message += f", test={len(test_samples)}"
    print(split_message)

    write_dataset_split_csv(settings["output_dir"] / "dataset_split.csv", splits)
    write_class_mapping(settings["output_dir"] / "class_to_idx.json", class_to_idx)

    train_loader = make_loader(train_samples, class_to_idx, settings, training=True, device=device)
    val_loader = make_loader(val_samples, class_to_idx, settings, training=False, device=device)

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
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = build_cosine_scheduler(optimizer, settings["epochs"])
    amp_enabled = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    selection_metric = settings["selection_metric"]
    valid_selection_metrics = {"val_loss", "val_top1", "val_top5", "val_macro_f1"}
    if selection_metric not in valid_selection_metrics:
        raise ValueError(
            f"Unsupported selection metric: {selection_metric}. "
            f"Expected one of {sorted(valid_selection_metrics)}"
        )

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    resolved_config = {
        "settings": serializable_settings(settings),
        "device": str(device),
        "gpu": gpu_name,
        "amp_enabled": amp_enabled,
        "num_classes": len(classes),
        "split_counts": {name: len(split_samples) for name, split_samples in splits.items()},
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameter_count,
    }
    resolved_config_path = settings["output_dir"] / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

    print(f"AMP enabled: {amp_enabled}")
    print(f"Selection metric: {selection_metric}")
    print(f"Parameters: total={total_parameters:,}, trainable={trainable_parameter_count:,}")

    best_score = float("inf") if selection_metric == "val_loss" else float("-inf")
    best_row: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    early_stopped = False
    start_time = time.time()
    best_model_path = settings["output_dir"] / f"best_{settings['model_name']}.pt"
    history_path = settings["output_dir"] / "history.csv"
    metrics_path = settings["output_dir"] / "metrics.csv"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, settings["epochs"] + 1):
        print(f"\nEpoch {epoch}/{settings['epochs']}")
        epoch_start = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=settings["max_batches"],
            scaler=scaler,
            amp_enabled=amp_enabled,
            num_classes=len(classes),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=settings["max_batches"],
            amp_enabled=amp_enabled,
            num_classes=len(classes),
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_top1": train_metrics.top1,
            "val_loss": val_metrics.loss,
            "val_top1": val_metrics.top1,
            "val_top5": val_metrics.top5,
            "val_macro_f1": val_metrics.macro_f1,
            "learning_rate": learning_rate,
            "epoch_time_seconds": time.time() - epoch_start,
        }
        history.append(row)
        write_history_csv(history_path, history)
        write_history_csv(metrics_path, history)
        print(
            f"train_loss={train_metrics.loss:.4f}, train_top1={train_metrics.top1:.4f}, "
            f"val_loss={val_metrics.loss:.4f}, val_top1={val_metrics.top1:.4f}, "
            f"val_top5={val_metrics.top5:.4f}, val_macro_f1={val_metrics.macro_f1:.4f}, "
            f"time={row['epoch_time_seconds']:.1f}s"
        )

        current_score = float(row[selection_metric])
        improved = current_score < best_score if selection_metric == "val_loss" else current_score > best_score
        if improved:
            best_score = current_score
            best_row = dict(row)
            epochs_without_improvement = 0
            save_checkpoint(
                best_model_path,
                model=model,
                class_to_idx=class_to_idx,
                settings=settings,
                best_val_acc=val_metrics.top1,
                epoch=epoch,
                best_metrics=best_row,
                selection_metric=selection_metric,
            )
            print(f"Saved best model to {best_model_path}")
        else:
            epochs_without_improvement += 1

        patience = settings["early_stopping_patience"]
        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            early_stopped = True
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    if best_row is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    elapsed_seconds = time.time() - start_time
    test_metrics = None
    if settings["evaluate_test"]:
        if test_samples is None:
            raise RuntimeError("Test evaluation was requested but no test split was loaded.")
        checkpoint = load_checkpoint(best_model_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loader = make_loader(test_samples, class_to_idx, settings, training=False, device=device)
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            max_batches=settings["max_batches"],
            amp_enabled=amp_enabled,
            num_classes=len(classes),
        )

    save_training_curves(settings["output_dir"], history)

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
        "test_evaluated": test_metrics is not None,
        "configured_epochs": settings["epochs"],
        "epochs_completed": len(history),
        "best_epoch": best_row["epoch"],
        "selection_metric": selection_metric,
        "best_selection_score": best_score,
        "best_val_loss": best_row["val_loss"],
        "minimum_val_loss": min(row["val_loss"] for row in history),
        "best_val_top1": best_row["val_top1"],
        "best_val_top5": best_row["val_top5"],
        "best_val_macro_f1": best_row["val_macro_f1"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameter_count,
        "amp_enabled": amp_enabled,
        "early_stopping_patience": settings["early_stopping_patience"],
        "early_stopped": early_stopped,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
        "average_epoch_seconds": sum(row["epoch_time_seconds"] for row in history) / len(history),
        "device": str(device),
        "gpu": gpu_name,
        "max_gpu_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "max_gpu_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
    }
    if test_metrics is not None:
        summary.update(
            {
                "test_images": len(test_samples),
                "test_loss": test_metrics.loss,
                "test_top1": test_metrics.top1,
                "test_top5": test_metrics.top5,
                "test_macro_f1": test_metrics.macro_f1,
            }
        )
    summary_path = settings["output_dir"] / "training_summary.json"
    summary_text = json.dumps(summary, indent=2)
    summary_path.write_text(summary_text, encoding="utf-8")
    (settings["output_dir"] / "summary.json").write_text(summary_text, encoding="utf-8")

    print("\nFinal result")
    print(summary_text)
    print(f"History saved to {history_path}")
    print(f"Summary saved to {summary_path}")


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(output_dir: Path, history: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MobileNetV2 loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    metric_curves = (
        ("val_top1", "Validation Top-1 accuracy", "val_top1_curve.png"),
        ("val_macro_f1", "Validation Macro-F1", "val_macro_f1_curve.png"),
        ("val_top5", "Validation Top-5 accuracy", "val_top5_curve.png"),
    )
    for metric, title, filename in metric_curves:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, [row[metric] for row in history], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.ylim(0.0, 1.0)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


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
    configured_patience = nested(config, "training", "early_stopping_patience")

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
        "amp": bool(pick(None, nested(config, "training", "amp"), False)),
        "early_stopping_patience": (
            int(configured_patience) if configured_patience is not None else None
        ),
        "selection_metric": str(
            pick(None, nested(config, "training", "selection_metric"), "val_top1")
        ),
        "evaluate_test": bool(pick(None, nested(config, "training", "evaluate_test"), True)),
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
