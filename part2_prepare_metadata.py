import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from src.data.build_metadata import (
    deduplicate_samples_by_content,
    parse_official_image_list,
    write_class_mapping,
)
from src.data.split_dataset import stratified_group_split, write_split_csvs


DEFAULT_DATA_ROOT = Path("data/raw/5061353/leafsnap-dataset-30subset")
OFFICIAL_LIST_NAME = "leafsnap-dataset-30subset-images.txt"


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_image_path(image_path: Path, data_root: Path) -> Path:
    try:
        return image_path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Image path is outside data_root: {image_path}") from exc


def find_official_list(data_root: Path, configured_path: Path | None) -> Path:
    if configured_path is not None:
        if not configured_path.exists():
            raise FileNotFoundError(f"Official image metadata list not found: {configured_path}")
        return configured_path

    candidates = [data_root / OFFICIAL_LIST_NAME, data_root.parent / OFFICIAL_LIST_NAME]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Official image metadata list not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def inspect_unique_images(
    samples: list[tuple[Path, str, str, str]], data_root: Path
) -> list[dict[str, object]]:
    quality_rows: list[dict[str, object]] = []
    errors: list[str] = []
    for image_path, label, source, content_hash in samples:
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                quality_rows.append(
                    {
                        "path": relative_image_path(image_path, data_root).as_posix(),
                        "label": label,
                        "source": source,
                        "sha256": content_hash,
                        "width": width,
                        "height": height,
                        "mode": image.mode,
                        "aspect_ratio": round(width / height, 4) if height else 0,
                    }
                )
        except Exception as exc:
            errors.append(f"{image_path}: {exc}")

    if errors:
        preview = "\n".join(errors[:10])
        raise RuntimeError(f"Found {len(errors)} missing or unreadable images:\n{preview}")
    return quality_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the locked, content-deduplicated Part 2 dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Leafsnap subset root containing dataset/; output CSV paths are relative to this directory.",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        default=None,
        help="Optional official image-list path. Defaults to data_root or its parent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not (data_root / "dataset" / "images").is_dir():
        raise FileNotFoundError(f"Leafsnap dataset/images directory not found under data_root: {data_root}")
    list_path = find_official_list(data_root, args.image_list)

    official_samples = parse_official_image_list(list_path, data_root)
    unique_samples, same_class_duplicates, conflicting_labels = deduplicate_samples_by_content(official_samples)

    conflict_hashes = {content_hash for content_hash, _, _, _ in conflicting_labels}
    unique_content_hashes = len(unique_samples) + len(conflict_hashes)
    if len(official_samples) != len(unique_samples) + len(same_class_duplicates) + len(conflicting_labels):
        raise RuntimeError("Content-deduplication counts do not reconcile to the official manifest.")

    classes = sorted({label for _, label, _, _ in unique_samples})
    if len(classes) != 30:
        raise RuntimeError(f"Expected 30 classes after cleaning, found {len(classes)}")
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    quality_rows = inspect_unique_images(unique_samples, data_root)
    canonical_samples = [
        (relative_image_path(image_path, data_root), label, source, content_hash)
        for image_path, label, source, content_hash in unique_samples
    ]

    metadata_dir = Path("data/metadata")
    split_dir = Path("data/splits")
    write_class_mapping(metadata_dir / "class_to_idx.json", class_to_idx)
    save_csv(
        metadata_dir / "images.csv",
        ["path", "label", "class_idx", "source", "sha256"],
        [
            {
                "path": image_path.as_posix(),
                "label": label,
                "class_idx": class_to_idx[label],
                "source": source,
                "sha256": content_hash,
            }
            for image_path, label, source, content_hash in canonical_samples
        ],
    )
    save_csv(
        metadata_dir / "duplicate_same_class.csv",
        ["sha256", "canonical_path", "canonical_source", "duplicate_path", "label", "source"],
        [
            {
                "sha256": content_hash,
                "canonical_path": relative_image_path(canonical_path, data_root).as_posix(),
                "canonical_source": canonical_source,
                "duplicate_path": relative_image_path(duplicate_path, data_root).as_posix(),
                "label": label,
                "source": duplicate_source,
            }
            for content_hash, canonical_path, canonical_source, duplicate_path, label, duplicate_source
            in same_class_duplicates
        ],
    )
    save_csv(
        metadata_dir / "conflicting_labels.csv",
        ["sha256", "path", "label", "source"],
        [
            {
                "sha256": content_hash,
                "path": relative_image_path(image_path, data_root).as_posix(),
                "label": label,
                "source": source,
            }
            for content_hash, image_path, label, source in conflicting_labels
        ],
    )

    split_samples = [(image_path, label) for image_path, label, _, _ in canonical_samples]
    train_samples, val_samples, test_samples = stratified_group_split(
        split_samples,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    expected_classes = set(classes)
    for split_name, subset in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_classes = {label for _, label in subset}
        if split_classes != expected_classes:
            missing_classes = sorted(expected_classes - split_classes)
            raise RuntimeError(f"{split_name} split is missing classes: {missing_classes}")
    write_split_csvs(split_dir, {"train": train_samples, "val": val_samples, "test": test_samples})

    label_counts: dict[str, Counter[str]] = {}
    for _, label, source, _ in canonical_samples:
        counter = label_counts.setdefault(label, Counter())
        counter[source] += 1
        counter["total"] += 1
    save_csv(
        metadata_dir / "dataset_summary.csv",
        ["label", "class_idx", "total", "field", "lab"],
        [
            {
                "label": label,
                "class_idx": class_to_idx[label],
                "total": label_counts[label]["total"],
                "field": label_counts[label]["field"],
                "lab": label_counts[label]["lab"],
            }
            for label in classes
        ],
    )

    split_rows: list[dict[str, object]] = []
    for split_name, subset in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_counter = Counter(label for _, label in subset)
        split_rows.extend(
            {"split": split_name, "label": label, "count": split_counter[label]}
            for label in classes
        )
    save_csv(metadata_dir / "split_audit.csv", ["split", "label", "count"], split_rows)
    save_csv(
        metadata_dir / "image_quality.csv",
        ["path", "label", "source", "sha256", "width", "height", "mode", "aspect_ratio"],
        quality_rows,
    )

    source_counts = Counter(source for _, _, source, _ in canonical_samples)
    quality_summary = {
        "official_manifest_samples": len(official_samples),
        "unique_content_hashes": unique_content_hashes,
        "same_class_duplicates_removed": len(same_class_duplicates),
        "conflicting_hash_groups_removed": len(conflict_hashes),
        "conflicting_images_removed": len(conflicting_labels),
        "total_samples": len(canonical_samples),
        "readable_images": len(quality_rows),
        "missing_or_invalid": 0,
        "field_images": source_counts["field"],
        "lab_images": source_counts["lab"],
        "class_count": len(classes),
        "mode_counts": dict(Counter(row["mode"] for row in quality_rows)),
    }
    (metadata_dir / "data_quality_summary.json").write_text(
        json.dumps(quality_summary, indent=2), encoding="utf-8"
    )

    print(f"Official manifest samples: {len(official_samples)}")
    print(f"Unique content hashes: {unique_content_hashes}")
    print(f"Removed same-class duplicate paths: {len(same_class_duplicates)}")
    print(f"Excluded conflicting hashes/images: {len(conflict_hashes)}/{len(conflicting_labels)}")
    print(f"Final samples/classes: {len(canonical_samples)}/{len(classes)}")
    print(f"Split sizes: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")


if __name__ == "__main__":
    main()
