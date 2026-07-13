from torch import nn


def build_classification_loss() -> nn.Module:
    return nn.CrossEntropyLoss()
