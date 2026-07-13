# COMP9444-Tree-Species-Classification
A COMP9444 group project for automatic tree species classification using deep learning and image recognition.

## Project Structure

```text
configs/        Experiment configuration files
data/           Metadata and split files; raw images stay local
notebooks/      Exploration and experiment notebooks
src/            Reusable Python package code
outputs/        Local experiment outputs, checkpoints, and metrics
report/         Report figures, tables, and references
presentation/   Slides and presentation figures
```

## Part 1: Custom CNN Baseline

Part 1 trains a custom CNN from scratch, without pretrained weights, to measure baseline tree species classification performance.

The runnable entry point is still:

```powershell
py -3.12 part1_custom_cnn.py
```

The actual implementation now lives under `src/`, with baseline settings in `configs/baseline.yaml`.

See `PART1_CUSTOM_CNN.md` for setup, training commands, and report notes.

## Part 2: Data Preparation and EDA

Part 2 covers dataset metadata preparation, split generation, and exploratory analysis for the Leafsnap 30-subset data.

Key files:

- `part2_prepare_metadata.py` — generate `data/metadata/*` and `data/splits/*`
- `part2_data_eda.py` — run EDA and save figures under `report/figures/data/`
- `PART2_DATA.md` — Part 2 handoff summary for next contributors

Run Part 2 scripts with:

```bash
python3 part2_prepare_metadata.py
python3 part2_data_eda.py
```
