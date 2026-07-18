from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mobilenetv2 import build_mobilenetv2
from src.training.trainer import (
    build_optimizer,
    macro_f1_from_confusion,
    run_epoch,
    save_training_curves,
    trainable_feature_block_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FixedLogitsModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_epoch_metrics_include_top5_and_macro_f1() -> None:
    logits = torch.tensor(
        [
            [10.0, 0.0, -1.0, -2.0, -3.0, -4.0],
            [-4.0, -3.0, -2.0, -1.0, 10.0, 9.0],
        ]
    )
    targets = torch.tensor([0, 5], dtype=torch.int64)
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2, shuffle=False)

    metrics = run_epoch(
        FixedLogitsModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        num_classes=6,
    )

    assert metrics.top1 == 0.5
    assert metrics.top5 == 1.0
    assert metrics.macro_f1 == 1.0 / 6.0
    assert metrics.loss > 0


def test_macro_f1_is_one_for_perfect_confusion_matrix() -> None:
    assert macro_f1_from_confusion(torch.eye(30, dtype=torch.int64)) == 1.0


def test_frozen_config_selects_macro_f1_without_test_evaluation() -> None:
    config_path = REPO_ROOT / "configs/mobilenetv2_frozen.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["training"]["selection_metric"] == "val_macro_f1"
    assert config["training"]["evaluate_test"] is False
    assert config["training"]["amp"] is True
    assert config["training"]["early_stopping_patience"] == 5
    assert config["training"]["epochs"] == 15


def test_partial_config_and_differential_optimizer_groups() -> None:
    config_path = REPO_ROOT / "configs/mobilenetv2_partial.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_mobilenetv2(
        num_classes=30,
        pretrained=False,
        fine_tune_mode="partial",
        unfreeze_blocks=4,
    )
    optimizer, groups = build_optimizer(
        model,
        {
            "learning_rate": config["training"]["learning_rate"],
            "feature_learning_rate": config["training"]["feature_learning_rate"],
            "classifier_learning_rate": config["training"]["classifier_learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
        },
    )

    assert config["training"]["evaluate_test"] is False
    assert config["training"]["selection_metric"] == "val_macro_f1"
    assert config["model"]["initial_checkpoint"].endswith("best_mobilenetv2_frozen.pt")
    assert trainable_feature_block_indices(model) == [15, 16, 17, 18]
    assert [group["name"] for group in groups] == ["features", "classifier"]
    assert [group["parameter_count"] for group in groups] == [1_526_080, 38_430]
    assert [group["initial_learning_rate"] for group in groups] == [1e-5, 1e-4]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]


def test_training_curves_are_written(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_loss": 1.1,
            "val_top1": 0.5,
            "val_top5": 0.9,
            "val_macro_f1": 0.4,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "val_loss": 0.9,
            "val_top1": 0.6,
            "val_top5": 0.95,
            "val_macro_f1": 0.5,
        },
    ]
    save_training_curves(tmp_path, history)

    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "val_top1_curve.png").is_file()
    assert (tmp_path / "val_top5_curve.png").is_file()
    assert (tmp_path / "val_macro_f1_curve.png").is_file()
