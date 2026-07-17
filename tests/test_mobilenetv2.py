from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.data.transforms import LeafImageTransform
from src.models.baseline_cnn import CustomCNN
from src.models.mobilenetv2 import MobileNetV2Transfer, build_mobilenetv2
from src.utils.checkpoint import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("mode", ["frozen", "partial", "full"])
def test_fine_tune_modes_select_expected_parameters(mode: str) -> None:
    unfreeze_blocks = 4
    model = build_mobilenetv2(
        num_classes=30,
        pretrained=False,
        fine_tune_mode=mode,  # type: ignore[arg-type]
        unfreeze_blocks=unfreeze_blocks,
    )
    model.train()

    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    if mode == "frozen":
        assert all(not parameter.requires_grad for parameter in model.features.parameters())
        assert all(not block.training for block in model.features)
    elif mode == "partial":
        split_index = model.num_feature_blocks - unfreeze_blocks
        assert all(
            not parameter.requires_grad
            for block in model.features[:split_index]
            for parameter in block.parameters()
        )
        assert all(
            parameter.requires_grad
            for block in model.features[split_index:]
            for parameter in block.parameters()
        )
        assert all(not block.training for block in model.features[:split_index])
        assert all(block.training for block in model.features[split_index:])
    else:
        assert all(parameter.requires_grad for parameter in model.parameters())
        assert all(block.training for block in model.features)

    counts = model.parameter_counts()
    assert counts.total > 0
    assert counts.trainable > 0
    assert counts.frozen == counts.total - counts.trainable
    assert (counts.trainable == counts.total) is (mode == "full")


def test_frozen_forward_backward_optimizer_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = MobileNetV2Transfer(num_classes=30, pretrained=False, fine_tune_mode="frozen")
    model.train()
    feature_parameter = next(model.features.parameters())
    classifier_parameter = model.classifier[-1].weight
    feature_before = feature_parameter.detach().clone()
    classifier_before = classifier_parameter.detach().clone()

    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.tensor([0, 29], dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    assert outputs.shape == (2, 30)
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.features.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())

    optimizer.step()
    assert torch.equal(feature_parameter, feature_before)
    assert not torch.equal(classifier_parameter, classifier_before)

    checkpoint_path = tmp_path / "mobilenetv2_smoke.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        class_to_idx={f"class_{index}": index for index in range(30)},
        settings={"fine_tune_mode": "frozen", "output_dir": tmp_path},
        best_val_acc=0.0,
    )
    restored = MobileNetV2Transfer(num_classes=30, pretrained=False, fine_tune_mode="frozen")
    checkpoint = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    restored.load_state_dict(checkpoint["model_state_dict"])
    assert all(
        torch.equal(model.state_dict()[key].cpu(), restored.state_dict()[key])
        for key in model.state_dict()
    )


def test_imagenet_normalization_is_selectable_without_changing_baseline_default() -> None:
    white_image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    baseline = LeafImageTransform(image_size=16, training=False, augment=False)(white_image)
    imagenet = LeafImageTransform(
        image_size=16,
        training=False,
        augment=False,
        normalization="imagenet",
    )(white_image)

    assert torch.allclose(baseline, torch.ones_like(baseline))
    expected = torch.tensor(
        [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
    ).view(3, 1, 1)
    assert torch.allclose(imagenet, expected.expand_as(imagenet))


def test_custom_cnn_interface_remains_compatible() -> None:
    model = CustomCNN(num_classes=30)
    outputs = model(torch.randn(2, 3, 128, 128))
    assert outputs.shape == (2, 30)
