from typing import Literal

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


TrainableScope = Literal["head", "last_blocks", "all"]


def build_efficientnet_b0(
    num_classes: int = 30,
    trainable_scope: TrainableScope = "head",
    pretrained: bool = True,
) -> nn.Module:
    """Build EfficientNet-B0 and configure which parameters are trainable.

    ``head`` trains only the classifier, ``last_blocks`` trains the final two
    feature blocks and classifier, and ``all`` fine-tunes the entire model.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if trainable_scope not in {"head", "last_blocks", "all"}:
        raise ValueError(
            "trainable_scope must be 'head', 'last_blocks', or 'all'"
        )

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if trainable_scope in {"last_blocks", "all"}:
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True

    if trainable_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return the total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
