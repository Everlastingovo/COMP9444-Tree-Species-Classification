from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


Normalization = Literal["standard", "imagenet"]

NORMALIZATION_STATS = {
    "standard": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def preprocess_evaluation_image(
    image: Image.Image,
    *,
    image_size: int,
    normalization: Normalization,
    brightness_factor: float = 1.0,
) -> torch.Tensor:
    """Apply the locked deterministic test transform and optional brightness."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if normalization not in NORMALIZATION_STATS:
        raise ValueError(f"Unsupported normalization: {normalization}")
    if brightness_factor <= 0:
        raise ValueError("brightness_factor must be positive.")

    transformed = image.convert("RGB")
    if brightness_factor != 1.0:
        transformed = ImageEnhance.Brightness(transformed).enhance(
            brightness_factor
        )
    transformed = ImageOps.contain(transformed, (image_size, image_size))
    transformed = pad_to_square(transformed)
    transformed = transformed.resize((image_size, image_size))

    array = np.asarray(transformed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean, std = NORMALIZATION_STATS[normalization]
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(image, ((side - width) // 2, (side - height) // 2))
    return padded
