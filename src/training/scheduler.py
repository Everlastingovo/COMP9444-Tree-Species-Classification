import torch


def build_cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
