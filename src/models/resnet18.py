from typing import Literal

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


TrainableScope = Literal["head", "layer4", "all"]


def build_resnet18(
    num_classes: int = 30,
    trainable_scope: TrainableScope = "head",
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet18 classifier and configure which layers are trainable.

    Args:
        num_classes: Number of output classes.
        trainable_scope: ``head`` trains only the final classifier,
            ``layer4`` trains the final residual block and classifier, and
            ``all`` fine-tunes the whole network.
        pretrained: Whether to load ImageNet pretrained weights.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if trainable_scope not in {"head", "layer4", "all"}:
        raise ValueError(
            "trainable_scope must be one of: 'head', 'layer4', or 'all'"
        )

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Start from a fully frozen model and selectively enable parameters below.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    if trainable_scope in {"layer4", "all"}:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    if trainable_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)`` for a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
