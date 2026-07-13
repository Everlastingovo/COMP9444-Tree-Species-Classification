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


def parse_official_image_list(list_path: Path, dataset_root: Path) -> list[tuple[Path, str, str]]:
    text = list_path.read_text(encoding="utf-8", errors="ignore")
    samples: list[tuple[Path, str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("file_id"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue

        image_path = Path(parts[1])
        source = parts[-1].lower()
        if source not in {"lab", "field"}:
            continue

        full_path = dataset_root / image_path
        if source == "lab" and "Auto_cropped" not in full_path.parts:
            continue
        if not full_path.exists():
            raise FileNotFoundError(f"Official image list referenced missing file: {full_path}")

        label = full_path.parent.name
        samples.append((full_path, label, source))
    return samples


def write_class_mapping(path: Path, class_to_idx: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")


def write_image_metadata(path: Path, samples: list[tuple], class_to_idx: dict[str, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        if class_to_idx is None:
            writer = csv.writer(csv_file)
            writer.writerow(["path", "label", "source"])
            for item in samples:
                if len(item) == 3:
                    image_path, label, source = item
                else:
                    image_path, label = item
                    source = image_path.parent.parent.name
                writer.writerow([str(image_path), label, source])
        else:
            writer = csv.writer(csv_file)
            writer.writerow(["path", "label", "class_idx", "source"])
            for item in samples:
                if len(item) == 3:
                    image_path, label, source = item
                else:
                    image_path, label = item
                    source = image_path.parent.parent.name
                writer.writerow([str(image_path), label, class_to_idx[label], source])
