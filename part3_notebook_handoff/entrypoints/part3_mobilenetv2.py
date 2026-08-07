from pathlib import Path
from typing import cast

from torch import nn

from src.models.mobilenetv2 import FineTuneMode, build_mobilenetv2
from src.training.trainer import main


def build_model(num_classes: int, settings: dict[str, object]) -> nn.Module:
    model = build_mobilenetv2(
        num_classes=num_classes,
        pretrained=bool(settings["pretrained"]),
        fine_tune_mode=cast(FineTuneMode, str(settings["fine_tune_mode"])),
        unfreeze_blocks=int(settings["unfreeze_blocks"]),
    )
    counts = model.parameter_counts()
    print(
        f"Parameters: total={counts.total:,}, trainable={counts.trainable:,}, "
        f"frozen={counts.frozen:,}"
    )
    return model


if __name__ == "__main__":
    main(
        model_factory=build_model,
        default_config=Path("configs/mobilenetv2_frozen.yaml"),
    )
