# COMP9444 Tree Species Classification

A COMP9444 group project for automatic tree-species classification using deep learning and image
recognition.

## Project structure

```text
configs/        Experiment configuration files
data/           Metadata and locked split files; raw images stay local
notebooks/      Exploration and experiment notebooks
src/            Reusable Python package code
outputs/        Local checkpoints and metrics
report/         Report figures, tables and references
presentation/   Slides and presentation figures
```

## Part 1: Custom CNN baseline

Part 1 trains a custom CNN from scratch without pretrained weights. The runnable entry point is:

```powershell
py -3.12 part1_custom_cnn.py
```

Implementation lives under `src/`, with settings in `configs/baseline.yaml`. See
`PART1_CUSTOM_CNN.md` for setup, training commands and report notes.

## Part 2: Data preparation and EDA

Part 2 builds metadata, content-level duplicate audits, a fixed split and exploratory analysis for
the Leafsnap 30-species subset.

The locked dataset contains 6,757 unique-content images across 30 classes. It is derived from 7,395
official manifest paths after removing 400 same-class duplicate paths and excluding 66 cross-class
SHA-256 conflict groups containing 238 images.

Key files:

- `part2_prepare_metadata.py`: generate `data/metadata/*` and `data/splits/*`.
- `part2_data_eda.py`: run EDA and save figures under `report/figures/data/`.
- `PART2_DATA.md`: Part 2 cleaning policy and handoff summary.
- `data/README.md`: data-root, audit and validation instructions.

The standard data location is `data/raw/5061353/leafsnap-dataset-30subset`. For an external local
copy, pass the subset root explicitly:

```powershell
python part2_prepare_metadata.py --data-root "D:\path\to\leafsnap-dataset-30subset"
python part2_data_eda.py --data-root "D:\path\to\leafsnap-dataset-30subset"
```

Do not regenerate the checked-in metadata or split files unless the team is intentionally relocking
the common data foundation.

## Part 3: MobileNetV2 transfer learning

Part 3 adds an ImageNet-pretrained MobileNetV2 with frozen, partial fine-tuning and full fine-tuning
modes. The scheduled experiments are the frozen and partial modes; both reuse the locked Part 2
splits and select ImageNet normalization without changing the CustomCNN defaults.

Use `part3_mobilenetv2.py` with `configs/mobilenetv2_frozen.yaml` or
`configs/mobilenetv2_partial.yaml`. Setup, smoke-test commands and experiment boundaries are in
`PART3_MOBILENETV2.md`.
