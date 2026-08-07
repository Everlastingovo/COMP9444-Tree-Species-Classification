from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


FineTuneMode = Literal["frozen", "partial", "full"]
VALID_FINE_TUNE_MODES = {"frozen", "partial", "full"}


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "trainable": self.trainable, "frozen": self.frozen}


class MobileNetV2Transfer(nn.Module):
    """MobileNetV2 adapted for configurable transfer learning."""

    def __init__(
        self,
        num_classes: int = 30,
        pretrained: bool = True,
        fine_tune_mode: FineTuneMode = "frozen",
        unfreeze_blocks: int = 4,
    ) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        self.features = backbone.features
        self.classifier = backbone.classifier
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.fine_tune_mode: FineTuneMode = fine_tune_mode
        self.unfreeze_blocks = unfreeze_blocks
        self._frozen_feature_indices: set[int] = set()
        self.configure_fine_tuning(fine_tune_mode, unfreeze_blocks)

    @property
    def num_feature_blocks(self) -> int:
        return len(self.features)

    @property
    def frozen_feature_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._frozen_feature_indices))

    def configure_fine_tuning(self, mode: FineTuneMode, unfreeze_blocks: int = 4) -> None:
        if mode not in VALID_FINE_TUNE_MODES:
            raise ValueError(
                f"Unsupported fine-tune mode: {mode}. "
                f"Expected one of {sorted(VALID_FINE_TUNE_MODES)}"
            )
        if mode == "partial" and not 1 <= unfreeze_blocks <= self.num_feature_blocks:
            raise ValueError(
                f"unfreeze_blocks must be between 1 and {self.num_feature_blocks}, got {unfreeze_blocks}"
            )

        for parameter in self.parameters():
            parameter.requires_grad = mode == "full"

        if mode in {"frozen", "partial"}:
            for parameter in self.classifier.parameters():
                parameter.requires_grad = True

        if mode == "partial":
            first_unfrozen = self.num_feature_blocks - unfreeze_blocks
            for block in self.features[first_unfrozen:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            self._frozen_feature_indices = set(range(first_unfrozen))
        elif mode == "frozen":
            self._frozen_feature_indices = set(range(self.num_feature_blocks))
        else:
            self._frozen_feature_indices = set()

        self.fine_tune_mode = mode
        self.unfreeze_blocks = unfreeze_blocks
        self.train(self.training)

    def train(self, mode: bool = True) -> "MobileNetV2Transfer":
        super().train(mode)
        if mode:
            # Frozen BatchNorm buffers must not update during transfer learning.
            for index in self._frozen_feature_indices:
                self.features[index].eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)

    def parameter_counts(self) -> ParameterCounts:
        return count_parameters(self)


def count_parameters(model: nn.Module) -> ParameterCounts:
    return ParameterCounts(
        total=sum(parameter.numel() for parameter in model.parameters()),
        trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def build_mobilenetv2(
    num_classes: int = 30,
    pretrained: bool = True,
    fine_tune_mode: FineTuneMode = "frozen",
    unfreeze_blocks: int = 4,
) -> MobileNetV2Transfer:
    return MobileNetV2Transfer(
        num_classes=num_classes,
        pretrained=pretrained,
        fine_tune_mode=fine_tune_mode,
        unfreeze_blocks=unfreeze_blocks,
    )
