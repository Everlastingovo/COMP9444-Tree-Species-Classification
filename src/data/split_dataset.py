import csv
import random
from pathlib import Path


def sample_group_id(image_path: Path) -> str:
    stem = image_path.stem
    if "-" in stem:
        prefix, tail = stem.rsplit("-", 1)
        if tail.isdigit():
            return prefix
    return stem


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


def stratified_group_split(
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
    for label, label_samples in by_label.items():
        group_to_samples: dict[str, list[tuple[Path, str]]] = {}
        for sample in label_samples:
            group = sample_group_id(sample[0])
            group_to_samples.setdefault(group, []).append(sample)

        groups = list(group_to_samples.values())
        rng.shuffle(groups)

        total = len(label_samples)
        target_train = max(1, round(total * (1 - val_ratio - test_ratio)))
        target_val = max(1, round(total * val_ratio))
        target_test = max(1, round(total * test_ratio))
        assigned = [0, 0, 0]
        targets = [target_train, target_val, target_test]

        for group_samples in groups:
            size = len(group_samples)
            remaining = [targets[i] - assigned[i] for i in range(3)]
            if any(r > 0 for r in remaining):
                best_split = max(
                    range(3),
                    key=lambda i: remaining[i] if remaining[i] > 0 else float("-inf"),
                )
            else:
                best_split = min(
                    range(3),
                    key=lambda i: abs((assigned[i] + size) - targets[i]),
                )
            if best_split == 0:
                train.extend(group_samples)
            elif best_split == 1:
                val.extend(group_samples)
            else:
                test.extend(group_samples)
            assigned[best_split] += size

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
