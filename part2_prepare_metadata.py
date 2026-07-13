from collections import Counter
from pathlib import Path
import csv

from src.data.build_metadata import parse_official_image_list, write_class_mapping, write_image_metadata
from src.data.split_dataset import stratified_group_split, write_split_csvs


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_root = Path("data/raw/5061353")
    subset_root = raw_root / "leafsnap-dataset-30subset"
    list_path = raw_root / "leafsnap-dataset-30subset-images.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Official image metadata list not found: {list_path}")

    samples = parse_official_image_list(list_path, subset_root)
    print(f"Loaded {len(samples)} official image records")

    classes = sorted({label for _, label, _ in samples})
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    write_class_mapping(Path("data/metadata/class_to_idx.json"), class_to_idx)
    write_image_metadata(Path("data/metadata/images.csv"), samples, class_to_idx)

    simple_samples = [(image_path, label) for image_path, label, _ in samples]
    train_samples, val_samples, test_samples = stratified_group_split(
        simple_samples,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    write_split_csvs(Path("data/splits"), {"train": train_samples, "val": val_samples, "test": test_samples})

    summary_rows = []
    label_groups: dict[str, Counter[str]] = {}
    for image_path, label, source in samples:
        counter = label_groups.setdefault(label, Counter())
        counter[source] += 1
        counter["total"] += 1

    for label in classes:
        counts = label_groups[label]
        summary_rows.append(
            {
                "label": label,
                "class_idx": class_to_idx[label],
                "total": counts["total"],
                "field": counts["field"],
                "lab": counts["lab"],
            }
        )

    save_csv(Path("data/metadata/dataset_summary.csv"), ["label", "class_idx", "total", "field", "lab"], summary_rows)

    split_counts = []
    for split_name, subset in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_counter = Counter(label for _, label in subset)
        for label in classes:
            split_counts.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": split_counter[label],
                }
            )
    save_csv(Path("data/metadata/split_audit.csv"), ["split", "label", "count"], split_counts)

    print("Wrote data/metadata/images.csv, class_to_idx.json, dataset_summary.csv, split_audit.csv")
    print("Wrote data/splits/train.csv, data/splits/val.csv, data/splits/test.csv")


if __name__ == "__main__":
    main()
