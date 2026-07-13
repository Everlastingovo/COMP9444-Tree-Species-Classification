from pathlib import Path


def ensure_figure_dir(path: Path = Path("report/figures")) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
