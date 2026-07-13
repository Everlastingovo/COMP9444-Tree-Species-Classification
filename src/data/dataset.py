from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import LeafImageTransform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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
        self.transform = LeafImageTransform(image_size=image_size, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return image, target
