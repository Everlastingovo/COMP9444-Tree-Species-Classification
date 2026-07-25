import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet18

from part5_export_baseline_predictions import load_checkpoint, load_mapping
from src.models.baseline_cnn import CustomCNN


@dataclass(frozen=True)
class LockedModel:
    name: str
    architecture: str
    checkpoint_path: Path
    checkpoint_sha256: str
    image_size: int
    normalization: str


def locked_models(repo_root: Path) -> list[LockedModel]:
    return [
        LockedModel(
            name="Baseline-CustomCNN",
            architecture="baseline",
            checkpoint_path=(
                repo_root
                / "outputs/baseline/member5_handoff/best_custom_cnn.pt"
            ),
            checkpoint_sha256=(
                "2ae3f85a28d46e2207c5cf6a8c37efe817af5794c8c2ea0845a6ebb02206367d"
            ),
            image_size=128,
            normalization="standard",
        ),
        LockedModel(
            name="MobileNetV2-Partial",
            architecture="mobilenetv2",
            checkpoint_path=(
                repo_root
                / "outputs/mobilenetv2/member5_handoff/checkpoint/"
                "best_mobilenetv2_partial.pt"
            ),
            checkpoint_sha256=(
                "9716e5f311abb5c828539f9ab7963c7444f6caf2cbc004e3c4c91838ddc9129a"
            ),
            image_size=224,
            normalization="imagenet",
        ),
        LockedModel(
            name="ResNet18-Layer4",
            architecture="resnet18",
            checkpoint_path=(
                repo_root
                / "outputs/resnet18/member5_handoff/checkpoint/best_model.pt"
            ),
            checkpoint_sha256=(
                "9ece5f02e59de428a475f5fe864cc4d4f97ec5dad3296fdb4d95244728aa0903"
            ),
            image_size=224,
            normalization="imagenet",
        ),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_locked_model(
    config: LockedModel,
    class_to_idx: dict[str, int],
) -> nn.Module:
    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")
    actual_hash = sha256_file(config.checkpoint_path)
    if actual_hash != config.checkpoint_sha256:
        raise ValueError(f"Checkpoint hash mismatch for {config.name}.")

    if config.architecture == "baseline":
        model = CustomCNN(num_classes=len(class_to_idx))
    elif config.architecture == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            len(class_to_idx),
        )
    elif config.architecture == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")

    checkpoint = load_checkpoint(config.checkpoint_path)
    checkpoint_mapping = {
        str(name): int(index)
        for name, index in checkpoint.get("class_to_idx", {}).items()
    }
    if checkpoint_mapping != class_to_idx:
        raise ValueError(f"Class mapping mismatch for {config.name}.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    configure_training_scope(model, config.architecture)
    return model.eval()


def configure_training_scope(model: nn.Module, architecture: str) -> None:
    if architecture == "baseline":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False
    if architecture == "mobilenetv2":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        for block in model.features[-4:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    elif architecture == "resnet18":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def benchmark_forward(
    model: nn.Module,
    *,
    image_size: int,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, float]:
    if batch_size <= 0 or measured_iterations <= 0 or warmup_iterations < 0:
        raise ValueError("Invalid benchmark iteration or batch-size setting.")
    inputs = torch.randn(batch_size, 3, image_size, image_size)

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(inputs)
        durations = []
        for _ in range(measured_iterations):
            start = time.perf_counter()
            model(inputs)
            durations.append(time.perf_counter() - start)

    median_batch_seconds = statistics.median(durations)
    mean_batch_seconds = statistics.mean(durations)
    return {
        "median_batch_ms": median_batch_seconds * 1000.0,
        "mean_batch_ms": mean_batch_seconds * 1000.0,
        "median_ms_per_image": median_batch_seconds * 1000.0 / batch_size,
        "mean_ms_per_image": mean_batch_seconds * 1000.0 / batch_size,
        "median_images_per_second": batch_size / median_batch_seconds,
    }


def run_efficiency_benchmark(
    *,
    repo_root: Path,
    class_mapping_path: Path,
    output_dir: Path,
    warmup_iterations: int,
    latency_iterations: int,
    throughput_iterations: int,
    throughput_batch_size: int,
) -> list[dict[str, Any]]:
    class_to_idx = load_mapping(class_mapping_path)
    rows = []
    for config in locked_models(repo_root):
        print(f"Benchmarking {config.name}...", flush=True)
        model = build_locked_model(config, class_to_idx)
        total_parameters, trainable_parameters = parameter_counts(model)
        latency = benchmark_forward(
            model,
            image_size=config.image_size,
            batch_size=1,
            warmup_iterations=warmup_iterations,
            measured_iterations=latency_iterations,
        )
        throughput = benchmark_forward(
            model,
            image_size=config.image_size,
            batch_size=throughput_batch_size,
            warmup_iterations=warmup_iterations,
            measured_iterations=throughput_iterations,
        )
        rows.append(
            {
                "model": config.name,
                "architecture": config.architecture,
                "image_size": config.image_size,
                "normalization": config.normalization,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "checkpoint_size_bytes": config.checkpoint_path.stat().st_size,
                "checkpoint_size_mb": (
                    config.checkpoint_path.stat().st_size / 1_000_000
                ),
                "batch1_median_latency_ms": latency["median_batch_ms"],
                "batch1_mean_latency_ms": latency["mean_batch_ms"],
                "batch1_images_per_second": latency[
                    "median_images_per_second"
                ],
                "batch_size_for_throughput": throughput_batch_size,
                "batch_throughput_median_ms_per_image": throughput[
                    "median_ms_per_image"
                ],
                "batch_throughput_images_per_second": throughput[
                    "median_images_per_second"
                ],
                "checkpoint_sha256": config.checkpoint_sha256,
            }
        )
        del model

    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows(output_dir / "model_efficiency.csv", rows)
    plot_efficiency(output_dir / "model_efficiency.png", rows)
    manifest = {
        "purpose": "Model-only inference efficiency on one shared CPU.",
        "device": "cpu",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "warmup_iterations": warmup_iterations,
        "latency_iterations": latency_iterations,
        "throughput_iterations": throughput_iterations,
        "throughput_batch_size": throughput_batch_size,
        "timing_scope": (
            "Forward pass only; excludes image decoding, preprocessing, and I/O."
        ),
        "input_policy": (
            "Each model uses its locked deployment resolution and normalization."
        ),
    }
    (output_dir / "efficiency_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return rows


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_efficiency(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["model"]) for row in rows]
    colors = ["#2A6F97", "#40916C", "#D97706"]
    parameter_values = [
        float(row["total_parameters"]) / 1_000_000 for row in rows
    ]
    latency_values = [
        float(row["batch1_median_latency_ms"]) for row in rows
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    parameter_bars = axes[0].bar(names, parameter_values, color=colors)
    axes[0].bar_label(parameter_bars, fmt="%.2fM", padding=3)
    axes[0].set_title("Model Parameters")
    axes[0].set_ylabel("Parameters (millions)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)

    latency_bars = axes[1].bar(names, latency_values, color=colors)
    axes[1].bar_label(latency_bars, fmt="%.2f ms", padding=3)
    axes[1].set_title("CPU Batch-1 Inference Latency")
    axes[1].set_ylabel("Median milliseconds per image")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark locked model size and CPU inference efficiency."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/metadata/class_to_idx.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison/experiments"),
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--throughput-iterations", type=int, default=15)
    parser.add_argument("--throughput-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_efficiency_benchmark(
        repo_root=args.repo_root.resolve(),
        class_mapping_path=args.class_mapping,
        output_dir=args.output_dir,
        warmup_iterations=args.warmup_iterations,
        latency_iterations=args.latency_iterations,
        throughput_iterations=args.throughput_iterations,
        throughput_batch_size=args.throughput_batch_size,
    )
    print(json.dumps(rows, indent=2))
    print(f"Saved efficiency outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
