import csv
import json
import zipfile
from pathlib import Path


def extract_dataset(zip_path: Path, extract_root: Path) -> Path:
    dataset_dir = extract_root / "leafsnap-dataset-30subset" / "dataset" / "images"
    if dataset_dir.exists():
        return dataset_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {zip_path}\n"
            "Download it from https://zenodo.org/records/5061353 and place it under data/raw/5061353/."
        )

    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting dataset from {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_root)
    return dataset_dir


def write_class_mapping(path: Path, class_to_idx: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")


def write_image_metadata(path: Path, samples: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "label", "source"])
        for image_path, label in samples:
            source = image_path.parent.parent.name
            writer.writerow([str(image_path), label, source])
