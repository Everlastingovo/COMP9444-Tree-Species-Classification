import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.dataset import LeafDataset, load_samples_from_csv
from src.data.transforms import LeafImageTransform
from src.models.mobilenetv2 import MobileNetV2Transfer, build_mobilenetv2
from src.utils.seed import set_seed


IMAGE_SIZE = 224
NORMALIZATION = "imagenet"
NUM_CLASSES = 30
EXPECTED_TEST_IMAGES = 1001
EXPECTED_GPU = "NVIDIA GeForce RTX 4060 Laptop GPU"
SELECTION_METRIC = "val_macro_f1"


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path
    epoch: int
    fine_tune_mode: str
    unfreeze_blocks: int = 4


@dataclass
class EvaluationResult:
    name: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    summary: dict[str, Any]
    class_report: list[dict[str, Any]]
    confusion: torch.Tensor
    source_metrics: dict[str, dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_class_mapping(path: Path) -> dict[str, int]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if len(mapping) != NUM_CLASSES or sorted(mapping.values()) != list(range(NUM_CLASSES)):
        raise ValueError("Class mapping must contain contiguous indices 0-29.")
    return {str(label): int(index) for label, index in mapping.items()}


def ordered_class_names(class_to_idx: dict[str, int]) -> list[str]:
    names = [""] * len(class_to_idx)
    for label, index in class_to_idx.items():
        names[index] = label
    return names


def validate_test_metadata(
    test_csv: Path,
    metadata_csv: Path,
    class_to_idx: dict[str, int],
) -> tuple[list[dict[str, str]], list[str]]:
    test_rows = read_csv(test_csv)
    if len(test_rows) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} test images, found {len(test_rows)}.")
    if {row["label"] for row in test_rows} != set(class_to_idx):
        raise ValueError("The test split does not contain exactly the configured 30 classes.")

    metadata_by_path = {row["path"]: row for row in read_csv(metadata_csv)}
    sources: list[str] = []
    for row in test_rows:
        metadata = metadata_by_path.get(row["path"])
        if metadata is None:
            raise ValueError(f"Test path is absent from metadata: {row['path']}")
        if metadata["label"] != row["label"]:
            raise ValueError(f"Label mismatch for test path: {row['path']}")
        if int(metadata["class_idx"]) != class_to_idx[row["label"]]:
            raise ValueError(f"Class-index mismatch for test path: {row['path']}")
        source = metadata["source"]
        if source not in {"lab", "field"}:
            raise ValueError(f"Unexpected source {source!r} for test path: {row['path']}")
        sources.append(source)
    return test_rows, sources


def validate_test_transform() -> LeafImageTransform:
    transform = LeafImageTransform(
        image_size=IMAGE_SIZE,
        training=False,
        augment=False,
        normalization=NORMALIZATION,
    )
    if transform.training or transform.augment:
        raise RuntimeError("Final test transform must not enable training or random augmentation.")
    synthetic = Image.new("RGB", (193, 271), (80, 120, 160))
    first = transform(synthetic)
    second = transform(synthetic)
    if first.shape != (3, IMAGE_SIZE, IMAGE_SIZE) or not torch.equal(first, second):
        raise RuntimeError("Final test transform is not deterministic at 224x224.")
    return transform


def checkpoint_manifest_entry(
    spec: CheckpointSpec,
    checkpoint_sha256: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    best_metrics = checkpoint.get("best_metrics") or {}
    return {
        "path": str(spec.path),
        "sha256": checkpoint_sha256,
        "checkpoint_epoch": spec.epoch,
        "checkpoint_recorded_epoch": checkpoint.get("epoch"),
        "selection_metric": SELECTION_METRIC,
        "selection_metric_description": "validation Macro-F1",
        "selection_score": best_metrics.get("val_macro_f1"),
        "strict_load": True,
        "test_evaluation_count": 0,
    }


def load_and_verify_checkpoint(
    spec: CheckpointSpec,
    class_to_idx: dict[str, int],
    device: torch.device = torch.device("cpu"),
) -> tuple[MobileNetV2Transfer, dict[str, Any]]:
    checkpoint = torch.load(spec.path, map_location="cpu")
    if checkpoint.get("epoch") != spec.epoch:
        raise ValueError(
            f"{spec.name} checkpoint epoch is {checkpoint.get('epoch')}, expected {spec.epoch}."
        )
    if checkpoint.get("selection_metric") != SELECTION_METRIC:
        raise ValueError(
            f"{spec.name} was not selected by validation Macro-F1: "
            f"{checkpoint.get('selection_metric')}"
        )
    if checkpoint.get("class_to_idx") != class_to_idx:
        raise ValueError(f"{spec.name} checkpoint class mapping does not match metadata.")
    if not bool((checkpoint.get("settings") or {}).get("pretrained")):
        raise ValueError(f"{spec.name} checkpoint does not record pretrained=True.")

    model = build_mobilenetv2(
        num_classes=NUM_CLASSES,
        pretrained=False,
        fine_tune_mode=spec.fine_tune_mode,  # type: ignore[arg-type]
        unfreeze_blocks=spec.unfreeze_blocks,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(
        torch.equal(value.detach().cpu(), checkpoint["model_state_dict"][key].detach().cpu())
        for key, value in model.state_dict().items()
    ):
        raise RuntimeError(f"{spec.name} parameters differ after strict checkpoint loading.")
    if model.classifier[-1].out_features != NUM_CLASSES:
        raise RuntimeError(f"{spec.name} classifier does not output {NUM_CLASSES} classes.")
    return model.to(device).eval(), checkpoint


def confusion_from_predictions(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    encoded = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_report_from_confusion(
    confusion: torch.Tensor,
    class_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    matrix = confusion.to(torch.float64)
    true_positives = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = torch.where(predicted > 0, true_positives / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positives / support, torch.zeros_like(support))
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator))
    supported = support > 0
    if not supported.any():
        raise ValueError("Cannot calculate a classification report without supported classes.")

    report = [
        {
            "class_idx": index,
            "class_name": class_names[index],
            "precision": precision[index].item(),
            "recall": recall[index].item(),
            "f1": f1[index].item(),
            "support": int(support[index].item()),
        }
        for index in range(len(class_names))
    ]
    macro = {
        "macro_precision": precision[supported].mean().item(),
        "macro_recall": recall[supported].mean().item(),
        "macro_f1": f1[supported].mean().item(),
        "classes_with_support": int(supported.sum().item()),
    }
    return report, macro


def metrics_from_outputs(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    top5_correct: torch.Tensor,
    losses: torch.Tensor,
    class_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    confusion = confusion_from_predictions(targets, predictions, len(class_names))
    report, macro = classification_report_from_confusion(confusion, class_names)
    metrics: dict[str, Any] = {
        "images": int(targets.numel()),
        "loss": losses.to(torch.float64).mean().item(),
        "top1": (predictions == targets).to(torch.float64).mean().item(),
        "top5": top5_correct.to(torch.float64).mean().item(),
        **macro,
    }
    return metrics, report, confusion


def source_breakdown(
    sources: list[str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    top5_correct: torch.Tensor,
    losses: torch.Tensor,
    class_names: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in ("lab", "field"):
        mask = torch.tensor([value == source for value in sources], dtype=torch.bool)
        metrics, _, _ = metrics_from_outputs(
            targets[mask],
            predictions[mask],
            top5_correct[mask],
            losses[mask],
            class_names,
        )
        result[source] = metrics
    return result


def evaluate_checkpoint_once(
    spec: CheckpointSpec,
    checkpoint_sha256: str,
    samples: list[tuple[Path, str]],
    sources: list[str],
    class_to_idx: dict[str, int],
    class_names: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
) -> EvaluationResult:
    model, _ = load_and_verify_checkpoint(spec, class_to_idx, device=device)
    dataset = LeafDataset(
        samples=samples,
        class_to_idx=class_to_idx,
        image_size=IMAGE_SIZE,
        training=False,
        augment=False,
        normalization=NORMALIZATION,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_top5_correct: list[torch.Tensor] = []
    all_losses: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                batch_losses = F.cross_entropy(logits, targets, reduction="none")
            predictions = logits.argmax(dim=1)
            top5 = logits.topk(k=5, dim=1).indices
            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())
            all_top5_correct.append((top5 == targets.unsqueeze(1)).any(dim=1).cpu())
            all_losses.append(batch_losses.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    top5_correct = torch.cat(all_top5_correct)
    losses = torch.cat(all_losses)
    if targets.numel() != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"{spec.name} evaluated {targets.numel()} images, expected {EXPECTED_TEST_IMAGES}."
        )
    summary, report, confusion = metrics_from_outputs(
        targets,
        predictions,
        top5_correct,
        losses,
        class_names,
    )
    summary.update(
        {
            "inference_seconds": inference_seconds,
            "average_inference_ms_per_image": inference_seconds * 1000.0 / targets.numel(),
        }
    )
    breakdown = source_breakdown(
        sources,
        targets,
        predictions,
        top5_correct,
        losses,
        class_names,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return EvaluationResult(
        name=spec.name,
        checkpoint_path=str(spec.path),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=spec.epoch,
        summary=summary,
        class_report=report,
        confusion=confusion,
        source_metrics=breakdown,
    )


def weakest_classes(report: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    return sorted(report, key=lambda row: (row["f1"], row["class_name"]))[:count]


def strongest_confusion_pairs(
    confusion: torch.Tensor,
    class_names: list[str],
    count: int = 10,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for first in range(len(class_names)):
        for second in range(first + 1, len(class_names)):
            first_as_second = int(confusion[first, second].item())
            second_as_first = int(confusion[second, first].item())
            total = first_as_second + second_as_first
            if total:
                pairs.append(
                    {
                        "class_a": class_names[first],
                        "class_b": class_names[second],
                        "a_predicted_as_b": first_as_second,
                        "b_predicted_as_a": second_as_first,
                        "total_confusions": total,
                    }
                )
    return sorted(
        pairs,
        key=lambda row: (-row["total_confusions"], row["class_a"], row["class_b"]),
    )[:count]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_classification_report(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["class_idx", "class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, confusion: torch.Tensor, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_class"] + class_names)
        for class_name, row in zip(class_names, confusion.tolist()):
            writer.writerow([class_name] + row)


def save_confusion_plot(path: Path, confusion: torch.Tensor, class_names: list[str], title: str) -> None:
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(confusion.numpy(), interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def metric_deltas(partial: dict[str, Any], frozen: dict[str, Any]) -> dict[str, float]:
    keys = (
        "loss",
        "top1",
        "top5",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "inference_seconds",
        "average_inference_ms_per_image",
    )
    return {key: float(partial[key] - frozen[key]) for key in keys}


def persist_results(
    output_dir: Path,
    results: list[EvaluationResult],
    class_names: list[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    result_rows = []
    result_by_name = {result.name: result for result in results}
    for result in results:
        result_rows.append(
            {
                "model": result.name,
                "checkpoint_path": result.checkpoint_path,
                "checkpoint_sha256": result.checkpoint_sha256,
                "checkpoint_epoch": result.checkpoint_epoch,
                **result.summary,
            }
        )
        write_classification_report(
            output_dir / f"{result.name}_classification_report.csv",
            result.class_report,
        )
        write_confusion_csv(
            output_dir / f"{result.name}_confusion_matrix.csv",
            result.confusion,
            class_names,
        )
        save_confusion_plot(
            output_dir / f"{result.name}_confusion_matrix.png",
            result.confusion,
            class_names,
            f"MobileNetV2 {result.name.title()} - Final Test Confusion Matrix",
        )

    with (output_dir / "final_test_results.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    frozen = result_by_name["frozen"]
    partial = result_by_name["partial"]
    summary = {
        "test_images": EXPECTED_TEST_IMAGES,
        "num_classes": NUM_CLASSES,
        "test_evaluated_once_per_checkpoint": True,
        "checkpoint_selection_used_test": False,
        "models": {
            result.name: {
                "checkpoint_path": result.checkpoint_path,
                "checkpoint_sha256": result.checkpoint_sha256,
                "checkpoint_epoch": result.checkpoint_epoch,
                "metrics": result.summary,
                "source_metrics": result.source_metrics,
                "weakest_classes_by_f1": weakest_classes(result.class_report),
                "strongest_confusion_pairs": strongest_confusion_pairs(
                    result.confusion,
                    class_names,
                ),
            }
            for result in results
        },
        "partial_minus_frozen": metric_deltas(partial.summary, frozen.summary),
    }
    write_json(output_dir / "final_test_summary.json", summary)
    (output_dir / "checkpoint_sha256.txt").write_text(
        "\n".join(
            f"{result.name}\t{result.checkpoint_sha256}\t{result.checkpoint_path}"
            for result in results
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "currently_evaluating": None,
            "test_only_evaluated_once": True,
        }
    )
    write_json(output_dir / "evaluation_manifest.json", manifest)
    return summary


def validate_environment(
    specs: list[CheckpointSpec],
    class_to_idx: dict[str, int],
    test_csv: Path,
    metadata_csv: Path,
    expected_gpu: str,
) -> tuple[dict[str, str], list[str], dict[str, dict[str, Any]]]:
    executable = Path(sys.executable).resolve()
    expected_venv = (Path.cwd() / ".venv" / "Scripts" / "python.exe").resolve()
    if executable != expected_venv:
        raise RuntimeError(f"Expected project .venv Python {expected_venv}, found {executable}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final test evaluation.")
    gpu = torch.cuda.get_device_name(0)
    if gpu != expected_gpu:
        raise RuntimeError(f"Expected GPU {expected_gpu!r}, found {gpu!r}.")

    _, sources = validate_test_metadata(test_csv, metadata_csv, class_to_idx)
    validate_test_transform()
    hashes: dict[str, str] = {}
    checkpoint_entries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if not spec.path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {spec.path}")
        checkpoint_hash = sha256_file(spec.path)
        hashes[spec.name] = checkpoint_hash
        model, checkpoint = load_and_verify_checkpoint(spec, class_to_idx)
        checkpoint_entries[spec.name] = checkpoint_manifest_entry(spec, checkpoint_hash, checkpoint)
        del model
    source_counts = {source: sources.count(source) for source in ("lab", "field")}
    preflight = {
        "sys_executable": str(executable),
        "cuda_available": str(torch.cuda.is_available()),
        "gpu": gpu,
        "test_images": str(EXPECTED_TEST_IMAGES),
        "num_classes": str(NUM_CLASSES),
        "image_size": str(IMAGE_SIZE),
        "normalization": NORMALIZATION,
        "random_augmentation": "False",
        "tta": "False",
        "ensemble": "False",
        "test_lab_images": str(source_counts["lab"]),
        "test_field_images": str(source_counts["field"]),
    }
    return preflight, sources, checkpoint_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time final MobileNetV2 test evaluation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata/images.csv"))
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/metadata/class_to_idx.json"),
    )
    parser.add_argument(
        "--frozen-checkpoint",
        type=Path,
        default=Path("outputs/mobilenetv2/frozen_seed42/best_mobilenetv2_frozen.pt"),
    )
    parser.add_argument(
        "--partial-checkpoint",
        type=Path,
        default=Path("outputs/mobilenetv2/partial_seed42/best_mobilenetv2_partial.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mobilenetv2/final_test"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-gpu", default=EXPECTED_GPU)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    class_to_idx = load_class_mapping(args.class_mapping)
    class_names = ordered_class_names(class_to_idx)
    specs = [
        CheckpointSpec("frozen", args.frozen_checkpoint, 15, "frozen"),
        CheckpointSpec("partial", args.partial_checkpoint, 14, "partial"),
    ]
    preflight, sources, checkpoint_entries = validate_environment(
        specs,
        class_to_idx,
        args.test_csv,
        args.metadata_csv,
        args.expected_gpu,
    )
    print(json.dumps({"preflight": preflight, "checkpoints": checkpoint_entries}, indent=2))
    if args.preflight_only:
        return
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite final-test output directory: {args.output_dir}"
        )

    test_rows, verified_sources = validate_test_metadata(
        args.test_csv,
        args.metadata_csv,
        class_to_idx,
    )
    if sources != verified_sources:
        raise RuntimeError("Test source metadata changed after preflight.")
    samples = load_samples_from_csv(args.test_csv, data_root=args.data_root)
    device = torch.device("cuda")
    amp_enabled = True
    manifest: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": preflight["sys_executable"],
        "device": "cuda",
        "gpu": preflight["gpu"],
        "test_split": str(args.test_csv),
        "test_images": len(test_rows),
        "num_classes": len(class_to_idx),
        "image_size": IMAGE_SIZE,
        "normalization": NORMALIZATION,
        "test_transform_training": False,
        "test_transform_augmentation": False,
        "shuffle": False,
        "amp_enabled": amp_enabled,
        "no_tta": True,
        "no_ensemble": True,
        "no_random_crop": True,
        "no_training_after_test": True,
        "checkpoint_selection_used_test": False,
        "checkpoints": checkpoint_entries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "evaluation_manifest.json", manifest)

    results: list[EvaluationResult] = []
    try:
        for spec in specs:
            manifest["currently_evaluating"] = spec.name
            write_json(args.output_dir / "evaluation_manifest.json", manifest)
            result = evaluate_checkpoint_once(
                spec=spec,
                checkpoint_sha256=checkpoint_entries[spec.name]["sha256"],
                samples=samples,
                sources=sources,
                class_to_idx=class_to_idx,
                class_names=class_names,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )
            results.append(result)
            checkpoint_entries[spec.name]["test_evaluation_count"] = 1
            write_json(args.output_dir / "evaluation_manifest.json", manifest)
        summary = persist_results(args.output_dir, results, class_names, manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(args.output_dir / "evaluation_manifest.json", manifest)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
