import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import LeafImageTransform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_DATA_ROOT = Path("data/raw/5061353/leafsnap-dataset-30subset")
DATA_ROOT_ENV = "LEAFSNAP_DATA_ROOT"


def collect_image_paths(images_root: Path, image_source: str) -> list[tuple[Path, str]]:
    sources = ["field", "lab"] if image_source == "all" else [image_source]
    samples: list[tuple[Path, str]] = []

    for source in sources:
        source_dir = images_root / source
        if not source_dir.exists():
            print(f"Warning: skipped missing image source: {source_dir}")
            continue

        class_dirs = []
        if source == "lab":
            auto_cropped = source_dir / "Auto_cropped"
            if auto_cropped.exists():
                class_dirs.extend(sorted(path for path in auto_cropped.iterdir() if path.is_dir()))
            else:
                print(f"Warning: lab source exists but Auto_cropped folder is missing: {auto_cropped}")
        else:
            class_dirs.extend(sorted(path for path in source_dir.iterdir() if path.is_dir()))

        for class_dir in class_dirs:
            label = class_dir.name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, label))

    if not samples:
        raise RuntimeError(f"No images found under {images_root} for source={image_source}")
    return samples


def resolve_data_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root)
    configured_root = os.environ.get(DATA_ROOT_ENV)
    return Path(configured_root) if configured_root else DEFAULT_DATA_ROOT


def resolve_image_path(image_path: Path, data_root: Path | None = None) -> Path:
    if image_path.is_absolute():
        return image_path
    return resolve_data_root(data_root) / image_path


def load_samples_from_csv(csv_path: Path, data_root: Path | None = None) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_path = resolve_image_path(Path(row["path"]), data_root)
            label = row["label"]
            samples.append((image_path, label))
    return samples


class LeafDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, str]],
        class_to_idx: dict[str, int],
        image_size: int,
        training: bool,
        augment: bool = True,
        normalization: str = "baseline",
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = LeafImageTransform(
            image_size=image_size,
            training=training,
            augment=augment,
            normalization=normalization,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return image, target
