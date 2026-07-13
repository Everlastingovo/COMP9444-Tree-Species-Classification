import argparse
import csv
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def extract_dataset(zip_path: Path, extract_root: Path) -> Path:
    dataset_dir = extract_root / "leafsnap-dataset-30subset" / "dataset" / "images"
    if dataset_dir.exists():
        return dataset_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {zip_path}\n"
            "Download it from https://zenodo.org/records/5061353 and place it under data/raw/5061353/."
        )

    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting dataset from {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_root)
    return dataset_dir


def collect_image_paths(images_root: Path, image_source: str) -> list[tuple[Path, str]]:
    sources = ["field", "lab"] if image_source == "all" else [image_source]
    samples: list[tuple[Path, str]] = []

    for source in sources:
        source_dir = images_root / source
        if not source_dir.exists():
            print(f"Warning: skipped missing image source: {source_dir}")
            continue

        for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            label = class_dir.name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, label))

    if not samples:
        raise RuntimeError(f"No images found under {images_root} for source={image_source}")
    return samples


def stratified_split(
    samples: list[tuple[Path, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    by_label: dict[str, list[tuple[Path, str]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n_total = len(label_samples)
        n_test = max(1, round(n_total * test_ratio))
        n_val = max(1, round(n_total * val_ratio))
        n_train = max(1, n_total - n_val - n_test)

        train.extend(label_samples[:n_train])
        val.extend(label_samples[n_train : n_train + n_val])
        test.extend(label_samples[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class LeafDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, str]],
        class_to_idx: dict[str, int],
        image_size: int,
        training: bool,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return image, target

    def transform(self, image: Image.Image) -> torch.Tensor:
        if self.training:
            image = ImageOps.contain(image, (self.image_size + 24, self.image_size + 24))
            image = pad_to_square(image)
            image = random_crop(image, self.image_size)
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.20))
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))
        else:
            image = ImageOps.contain(image, (self.image_size, self.image_size))
            image = pad_to_square(image)
            image = image.resize((self.image_size, self.image_size))

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(image, ((side - width) // 2, (side - height) // 2))
    return padded


def random_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        image = image.resize((max(width, size), max(height, size)))
        width, height = image.size
    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    return image.crop((left, top, left + size, top + size))


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


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


def save_split_csv(path: Path, splits: dict[str, list[tuple[Path, str]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "path", "label"])
        for split_name, samples in splits.items():
            for image_path, label in samples:
                writer.writerow([split_name, str(image_path), label])


def serializable_args(args: argparse.Namespace) -> dict[str, str | int | float | None]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1: train a custom CNN from scratch on Leafsnap.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Optional path to the extracted dataset/images folder. Use this if the zip was extracted elsewhere.",
    )
    parser.add_argument("--zip-path", type=Path, default=Path("data/raw/5061353/leafsnap-dataset-30subset.zip"))
    parser.add_argument("--extract-root", type=Path, default=Path("data/raw/5061353"))
    parser.add_argument("--image-source", choices=["field", "lab", "all"], default="field")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/part1_custom_cnn"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None, help="Use a small number for a quick smoke test.")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_root = args.images_root if args.images_root is not None else extract_dataset(args.zip_path, args.extract_root)
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    samples = collect_image_paths(images_root, args.image_source)
    classes = sorted({label for _, label in samples})
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    train_samples, val_samples, test_samples = stratified_split(samples, args.val_ratio, args.test_ratio, args.seed)

    print(f"Device: {device}")
    print(f"Image source: {args.image_source}")
    print(f"Classes: {len(classes)}")
    print(f"Images: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    save_split_csv(
        args.output_dir / "dataset_split.csv",
        {"train": train_samples, "val": val_samples, "test": test_samples},
    )
    (args.output_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        LeafDataset(train_samples, class_to_idx, args.image_size, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        LeafDataset(val_samples, class_to_idx, args.image_size, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        LeafDataset(test_samples, class_to_idx, args.image_size, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CustomCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []
    start_time = time.time()
    best_model_path = args.output_dir / "best_custom_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, max_batches=args.max_batches
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "args": serializable_args(args),
                    "best_val_acc": best_val_acc,
                },
                best_model_path,
            )
            print(f"Saved best model to {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, max_batches=args.max_batches)

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": "Custom CNN trained from scratch",
        "image_source": args.image_source,
        "num_classes": len(classes),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "test_images": len(test_samples),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_minutes": (time.time() - start_time) / 60,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal result")
    print(json.dumps(summary, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
