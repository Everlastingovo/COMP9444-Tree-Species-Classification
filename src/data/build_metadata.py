import csv
import hashlib
import json
import zipfile
from collections import defaultdict
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

        if source == "lab" and "Auto_cropped" not in image_path.parts:
            lab_idx = image_path.parts.index("lab")
            image_path = Path(*image_path.parts[: lab_idx + 1], "Auto_cropped", *image_path.parts[lab_idx + 1 :])

        full_path = dataset_root / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Official image list referenced missing file: {full_path}")

        label = full_path.parent.name
        samples.append((full_path, label, source))
    return samples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_samples_by_content(
    samples: list[tuple[Path, str, str]],
) -> tuple[
    list[tuple[Path, str, str, str]],
    list[tuple[str, Path, str, Path, str, str]],
    list[tuple[str, Path, str, str]],
]:
    """Deduplicate official samples while preserving manifest order.

    The first path in a same-label hash group is the canonical sample. Hash
    groups containing more than one label are excluded completely and
    returned for manual audit rather than having a label guessed in code.
    """
    groups: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for image_path, label, source in samples:
        groups[sha256_file(image_path)].append((image_path, label, source))

    unique_samples: list[tuple[Path, str, str, str]] = []
    same_class_duplicates: list[tuple[str, Path, str, Path, str, str]] = []
    conflicting_labels: list[tuple[str, Path, str, str]] = []

    for content_hash, group in groups.items():
        labels = {label for _, label, _ in group}
        if len(labels) > 1:
            conflicting_labels.extend(
                (content_hash, image_path, label, source)
                for image_path, label, source in group
            )
            continue

        canonical_path, label, canonical_source = group[0]
        unique_samples.append((canonical_path, label, canonical_source, content_hash))
        same_class_duplicates.extend(
            (content_hash, canonical_path, canonical_source, duplicate_path, label, duplicate_source)
            for duplicate_path, _, duplicate_source in group[1:]
        )

    return unique_samples, same_class_duplicates, conflicting_labels


def write_class_mapping(path: Path, class_to_idx: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")


def write_image_metadata(path: Path, samples: list[tuple], class_to_idx: dict[str, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        if class_to_idx is None:
            writer = csv.writer(csv_file, lineterminator="\n")
            writer.writerow(["path", "label", "source"])
            for item in samples:
                if len(item) == 3:
                    image_path, label, source = item
                else:
                    image_path, label = item
                    source = image_path.parent.parent.name
                writer.writerow([str(image_path), label, source])
        else:
            writer = csv.writer(csv_file, lineterminator="\n")
            writer.writerow(["path", "label", "class_idx", "source"])
            for item in samples:
                if len(item) == 3:
                    image_path, label, source = item
                else:
                    image_path, label = item
                    source = image_path.parent.parent.name
                writer.writerow([str(image_path), label, class_to_idx[label], source])
