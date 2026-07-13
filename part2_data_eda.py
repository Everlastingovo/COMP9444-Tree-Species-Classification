import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_metadata(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(headers):
                continue
            row = dict(zip(headers, parts))
            rows.append(row)
    return rows


def inspect_images(samples: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    missing = []
    for sample in samples:
        image_path = Path(sample["path"])
        if not image_path.exists():
            missing.append(str(image_path))
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
                rows.append(
                    {
                        "path": str(image_path),
                        "label": sample["label"],
                        "source": sample["source"],
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "aspect_ratio": round(width / height, 4) if height else 0,
                    }
                )
        except Exception as exc:
            missing.append(f"{image_path}: {exc}")
    if missing:
        print(f"{len(missing)} missing or unreadable images")
    return rows


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            values = [str(row.get(field, "")) for field in fieldnames]
            f.write(",".join(values) + "\n")


def plot_class_distribution(counts: Counter, path: Path) -> None:
    labels, values = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Class Distribution")
    plt.xlabel("Species")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_source_distribution(counts: Counter, path: Path) -> None:
    labels, values = zip(*counts.items())
    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%.1f%%", startangle=140, colors=["#5b8ff9", "#61dDAA"])
    plt.title("Source Distribution: field vs lab")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_image_size_distribution(sizes: list[tuple[int, int]], path: Path) -> None:
    widths, heights = zip(*sizes)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=40, color="#5b8ff9")
    plt.title("Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=40, color="#61dDAA")
    plt.title("Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Count")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_sample_pairs(samples: list[dict[str, object]], path: Path, max_examples: int = 8) -> None:
    examples = []
    fields = [s for s in samples if s["source"] == "field"]
    labs = [s for s in samples if s["source"] == "lab"]
    for group, subset in [("field", fields), ("lab", labs)]:
        subset_sorted = sorted(subset, key=lambda s: s["label"])[: max_examples // 2]
        examples.extend(subset_sorted)
    if not examples:
        return

    cols = 4
    rows = (len(examples) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, sample in enumerate(examples, start=1):
        with Image.open(sample["path"]) as image:
            image = image.convert("RGB")
            plt.subplot(rows, cols, i)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"{sample['source']}\n{sample['label']}", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_augmentation_examples(path: Path, image_path: Path, image_size: int = 128) -> None:
    from src.data.transforms import LeafImageTransform

    transform = LeafImageTransform(image_size=image_size, training=True, augment=True)
    with Image.open(image_path).convert("RGB") as image:
        figures = [transform(image.copy()) for _ in range(8)]
    plt.figure(figsize=(16, 8))
    for i, tensor in enumerate(figures, start=1):
        arr = ((tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        plt.subplot(2, 4, i)
        plt.imshow(arr)
        plt.axis("off")
        plt.title(f"Aug {i}", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    metadata_path = Path("data/metadata/images.csv")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    samples = load_metadata(metadata_path)
    print(f"Loaded {len(samples)} metadata rows")

    inspected = inspect_images(samples)
    print(f"Inspected {len(inspected)} readable images")

    source_counts = Counter(sample["source"] for sample in samples)
    label_counts = Counter(sample["label"] for sample in samples)
    mode_counts = Counter(sample["mode"] for sample in inspected)
    size_counts = [(sample["width"], sample["height"]) for sample in inspected]

    figures_dir = Path("report/figures/data")
    plot_class_distribution(label_counts, figures_dir / "class_distribution.png")
    plot_source_distribution(source_counts, figures_dir / "source_distribution.png")
    plot_image_size_distribution(size_counts, figures_dir / "image_size_distribution.png")
    plot_sample_pairs(inspected, figures_dir / "lab_vs_field_examples.png")
    if inspected:
        plot_augmentation_examples(figures_dir / "augmentation_examples.png", Path(inspected[0]["path"]))

    quality_rows = [
        {
            "path": row["path"],
            "label": row["label"],
            "source": row["source"],
            "width": row["width"],
            "height": row["height"],
            "mode": row["mode"],
            "aspect_ratio": row["aspect_ratio"],
        }
        for row in inspected
    ]
    save_csv(Path("data/metadata/image_quality.csv"), quality_rows, ["path", "label", "source", "width", "height", "mode", "aspect_ratio"])

    summary = {
        "total_samples": len(samples),
        "readable_images": len(inspected),
        "missing_or_invalid": len(samples) - len(inspected),
        "field_images": source_counts.get("field", 0),
        "lab_images": source_counts.get("lab", 0),
        "class_count": len(label_counts),
        "mode_counts": dict(mode_counts),
    }
    summary_path = Path("data/metadata/data_quality_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote image quality CSV and summary")


if __name__ == "__main__":
    main()
