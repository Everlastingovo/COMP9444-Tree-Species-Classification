from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class GradCAM:
    """Generate class activation maps for any model containing Conv2d layers."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer or find_last_conv_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

    def generate(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized heatmaps, logits, and the explained class indices."""
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width].")
        if images.shape[0] == 0:
            raise ValueError("images cannot be empty.")

        self.activations = None
        self.gradients = None
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Requiring input gradients keeps Grad-CAM working when a backbone is frozen.
        grad_images = images.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(grad_images)
            if logits.ndim != 2 or logits.shape[0] != images.shape[0]:
                raise ValueError("model output must have shape [batch, classes].")

            selected_classes = self._prepare_class_indices(logits, class_indices)
            selected_scores = logits.gather(1, selected_classes.unsqueeze(1)).sum()
            selected_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmaps = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmaps = F.relu(heatmaps)
        heatmaps = F.interpolate(
            heatmaps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        heatmaps = _normalize_heatmaps(heatmaps)

        return (
            heatmaps.detach().cpu(),
            logits.detach().cpu(),
            selected_classes.detach().cpu(),
        )

    def close(self) -> None:
        self._forward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _save_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return a tensor.")
        self.activations = output
        if not output.requires_grad:
            raise RuntimeError("Grad-CAM target activations do not require gradients.")
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    @staticmethod
    def _prepare_class_indices(
        logits: torch.Tensor,
        class_indices: torch.Tensor | Sequence[int] | None,
    ) -> torch.Tensor:
        if class_indices is None:
            return logits.argmax(dim=1)

        indices = torch.as_tensor(class_indices, dtype=torch.long, device=logits.device)
        if indices.ndim == 0:
            indices = indices.repeat(logits.shape[0])
        if indices.ndim != 1 or indices.numel() != logits.shape[0]:
            raise ValueError("class_indices must contain one class for every image.")
        if indices.min().item() < 0 or indices.max().item() >= logits.shape[1]:
            raise ValueError("class_indices contain a value outside the model output range.")
        return indices


def find_last_conv_layer(model: nn.Module) -> nn.Conv2d:
    """Find the final Conv2d module in Baseline, MobileNetV2, or ResNet18."""
    last_layer: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_layer = module
    if last_layer is None:
        raise ValueError("model does not contain a Conv2d layer.")
    return last_layer


def overlay_heatmap(
    image: Image.Image | np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if isinstance(image, Image.Image):
        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    else:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("image array must have shape [height, width, 3].")
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        rgb_image = Image.fromarray(
            np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
        )

    heatmap_array = np.asarray(
        heatmap.detach().cpu() if isinstance(heatmap, torch.Tensor) else heatmap,
        dtype=np.float32,
    )
    if heatmap_array.ndim != 2:
        raise ValueError("heatmap must have shape [height, width].")
    if heatmap_array.shape != image_array.shape[:2]:
        heatmap_image = Image.fromarray(
            np.clip(heatmap_array * 255.0, 0, 255).astype(np.uint8)
        )
        heatmap_image = heatmap_image.resize(rgb_image.size, Image.Resampling.BILINEAR)
        heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    colored_heatmap = colormaps["jet"](np.clip(heatmap_array, 0.0, 1.0))[..., :3]
    blended = (1.0 - alpha) * image_array + alpha * colored_heatmap
    return Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8))


def save_gradcam_grid(
    output_path: Path,
    original_images: Sequence[Image.Image],
    heatmaps: torch.Tensor,
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    confidences: Sequence[float],
) -> None:
    """Save paired original and Grad-CAM images for report or presentation use."""
    count = len(original_images)
    if count == 0:
        raise ValueError("at least one image is required.")
    if not (
        heatmaps.shape[0]
        == len(true_labels)
        == len(predicted_labels)
        == len(confidences)
        == count
    ):
        raise ValueError("all Grad-CAM inputs must contain the same number of items.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(count, 2, figsize=(8, count * 3.5), squeeze=False)
    for index, image in enumerate(original_images):
        original = image.convert("RGB")
        overlay = overlay_heatmap(original, heatmaps[index])
        label = (
            f"True: {true_labels[index]} | Pred: {predicted_labels[index]} "
            f"({confidences[index]:.1%})"
        )
        axes[index, 0].imshow(original)
        axes[index, 0].set_title(f"Original\n{label}", fontsize=9)
        axes[index, 1].imshow(overlay)
        axes[index, 1].set_title("Grad-CAM", fontsize=9)
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _normalize_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    flattened = heatmaps.flatten(start_dim=1)
    minimum = flattened.min(dim=1).values[:, None, None]
    maximum = flattened.max(dim=1).values[:, None, None]
    denominator = maximum - minimum
    return torch.where(
        denominator > 0,
        (heatmaps - minimum) / denominator,
        torch.zeros_like(heatmaps),
    )
