import csv
import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.build_metadata import deduplicate_samples_by_content, sha256_file
from src.data.dataset import LeafDataset, load_samples_from_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/raw/5061353/leafsnap-dataset-30subset"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def configured_data_root() -> Path:
    return Path(os.environ.get("LEAFSNAP_DATA_ROOT", DEFAULT_DATA_ROOT))


def require_data_root() -> Path:
    data_root = configured_data_root()
    if not (data_root / "dataset" / "images").is_dir():
        pytest.skip(f"Leafsnap data root is unavailable: {data_root}")
    return data_root


def test_deduplication_keeps_first_same_label_and_excludes_conflicts(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    conflict_a = tmp_path / "conflict_a.jpg"
    conflict_b = tmp_path / "conflict_b.jpg"
    unique = tmp_path / "unique.jpg"
    canonical.write_bytes(b"same-label-content")
    duplicate.write_bytes(b"same-label-content")
    conflict_a.write_bytes(b"conflicting-content")
    conflict_b.write_bytes(b"conflicting-content")
    unique.write_bytes(b"unique-content")

    samples = [
        (canonical, "class_a", "lab"),
        (duplicate, "class_a", "lab"),
        (conflict_a, "class_a", "field"),
        (conflict_b, "class_b", "field"),
        (unique, "class_b", "field"),
    ]
    cleaned, duplicates, conflicts = deduplicate_samples_by_content(samples)

    assert [row[0] for row in cleaned] == [canonical, unique]
    assert len(duplicates) == 1
    assert duplicates[0][1] == canonical
    assert duplicates[0][3] == duplicate
    assert {row[1] for row in conflicts} == {conflict_a, conflict_b}


def test_cleaned_metadata_and_audits_are_consistent() -> None:
    metadata = read_csv(REPO_ROOT / "data/metadata/images.csv")
    duplicates = read_csv(REPO_ROOT / "data/metadata/duplicate_same_class.csv")
    conflicts = read_csv(REPO_ROOT / "data/metadata/conflicting_labels.csv")
    with (REPO_ROOT / "data/metadata/class_to_idx.json").open(encoding="utf-8") as mapping_file:
        class_to_idx = json.load(mapping_file)

    assert len(metadata) == 6757
    assert len(duplicates) == 400
    assert len(conflicts) == 238
    assert len({row["sha256"] for row in conflicts}) == 66
    assert len(class_to_idx) == 30
    assert sorted(class_to_idx.values()) == list(range(30))
    assert len({row["path"] for row in metadata}) == len(metadata)
    assert len({row["sha256"] for row in metadata}) == len(metadata)
    assert all(not Path(row["path"]).is_absolute() for row in metadata)
    assert all(
        row["source"] != "lab" or "/lab/Auto_cropped/" in f"/{row['path']}"
        for row in metadata
    )

    kept_paths = {row["path"] for row in metadata}
    assert all(row["canonical_path"] in kept_paths for row in duplicates)
    assert all(row["duplicate_path"] not in kept_paths for row in duplicates)
    assert all(row["path"] not in kept_paths for row in conflicts)


def test_split_paths_and_content_are_disjoint_and_complete() -> None:
    metadata = read_csv(REPO_ROOT / "data/metadata/images.csv")
    metadata_by_path = {row["path"]: row for row in metadata}
    splits = {
        split: read_csv(REPO_ROOT / f"data/splits/{split}.csv")
        for split in ("train", "val", "test")
    }

    path_sets = {split: {row["path"] for row in rows} for split, rows in splits.items()}
    hash_sets = {
        split: {metadata_by_path[row["path"]]["sha256"] for row in rows}
        for split, rows in splits.items()
    }
    assert path_sets["train"].isdisjoint(path_sets["val"])
    assert path_sets["train"].isdisjoint(path_sets["test"])
    assert path_sets["val"].isdisjoint(path_sets["test"])
    assert hash_sets["train"].isdisjoint(hash_sets["val"])
    assert hash_sets["train"].isdisjoint(hash_sets["test"])
    assert hash_sets["val"].isdisjoint(hash_sets["test"])
    assert set().union(*path_sets.values()) == set(metadata_by_path)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 4734,
        "val": 1022,
        "test": 1001,
    }
    assert all(len({row["label"] for row in rows}) == 30 for rows in splits.values())


def test_all_metadata_hashes_match_files() -> None:
    data_root = require_data_root()
    metadata = read_csv(REPO_ROOT / "data/metadata/images.csv")
    for row in metadata:
        image_path = data_root / row["path"]
        assert image_path.is_file()
        assert sha256_file(image_path) == row["sha256"]


def test_dataloader_reads_224_batch() -> None:
    data_root = require_data_root()
    with (REPO_ROOT / "data/metadata/class_to_idx.json").open(encoding="utf-8") as mapping_file:
        class_to_idx = json.load(mapping_file)
    samples = load_samples_from_csv(REPO_ROOT / "data/splits/train.csv", data_root=data_root)
    dataset = LeafDataset(samples[:8], class_to_idx, image_size=224, training=True, augment=True)
    images, labels = next(iter(DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)))

    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 29
    assert torch.isfinite(images).all()
