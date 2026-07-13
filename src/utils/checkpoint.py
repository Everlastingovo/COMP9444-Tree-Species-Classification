from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: dict[str, int],
    settings: dict[str, Any],
    best_val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "settings": serializable_settings(settings),
            "best_val_acc": best_val_acc,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def serializable_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in settings.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result
