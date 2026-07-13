import csv
import random
from pathlib import Path


def stratified_split(
    samples: list[tuple[Path, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    by_label: dict[str, list[tuple[Path, str]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n_total = len(label_samples)
        n_test = max(1, round(n_total * test_ratio))
        n_val = max(1, round(n_total * val_ratio))
        n_train = max(1, n_total - n_val - n_test)

        train.extend(label_samples[:n_train])
        val.extend(label_samples[n_train : n_train + n_val])
        test.extend(label_samples[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_dataset_split_csv(path: Path, splits: dict[str, list[tuple[Path, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "path", "label"])
        for split_name, samples in splits.items():
            for image_path, label in samples:
                writer.writerow([split_name, str(image_path), label])


def write_split_csvs(split_dir: Path, splits: dict[str, list[tuple[Path, str]]]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        path = split_dir / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["path", "label"])
            for image_path, label in samples:
                writer.writerow([str(image_path), label])
