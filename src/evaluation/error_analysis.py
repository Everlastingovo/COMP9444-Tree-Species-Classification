from pathlib import Path


def top_errors(rows: list[dict], limit: int = 20) -> list[dict]:
    errors = [row for row in rows if row.get("label") != row.get("prediction")]
    return errors[:limit]


def save_error_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    import csv

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
