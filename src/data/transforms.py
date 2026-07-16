import random
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


Normalization = Literal["standard", "imagenet"]

NORMALIZATION_STATS = {
    "standard": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class LeafImageTransform:
    def __init__(
        self,
        image_size: int,
        training: bool,
        augment: bool = True,
        normalization: Normalization = "standard",
    ) -> None:
        if normalization not in NORMALIZATION_STATS:
            raise ValueError("normalization must be either 'standard' or 'imagenet'")

        self.image_size = image_size
        self.training = training
        self.augment = augment
        mean, std = NORMALIZATION_STATS[normalization]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.training and self.augment:
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
        return (tensor - self.mean) / self.std


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
