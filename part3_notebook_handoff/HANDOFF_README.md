# Part 3 MobileNetV2 Notebook handoff

This package contains the verified facts and artifacts needed to write or integrate the Part 3
Notebook. It intentionally excludes raw Leafsnap images, `.venv`, caches and intermediate
checkpoints.

## Start here

1. Read `notebook_outline.md` for the recommended section order.
2. Use `notebook_results_tables.csv` for validation/test result cells.
3. Use `experiments/frozen/` and `experiments/partial/` for histories and curves.
4. Use `final_test/` for per-class reports and confusion matrices.
5. Use `data_summary.md`, `model_method_summary.md` and `evaluation_protocol.md` for concise method
   descriptions.
6. Use `contribution_statement.md` for the Member 3 attribution section.

## Directory guide

```text
configs/             Frozen and Partial YAML configurations
checkpoints/         Only the two validation-selected best checkpoints
data/                Authoritative class mapping and lightweight dataset/split statistics
docs/                Original project, data and Part 3 documentation
entrypoints/         Training and final-evaluation entry points for reference
experiments/         Resolved configs, selected-run summaries, histories and four curves per mode
final_test/          Locked aggregate test outputs, reports, matrices, plots and manifest
src/                 Model, transforms, dataset, Trainer and supporting implementation files
```

## Recommended model

Use `checkpoints/best_mobilenetv2_partial.pt` for subsequent fixed analysis or XAI. It has SHA-256:

```text
9716e5f311abb5c828539f9ab7963c7444f6caf2cbc004e3c4c91838ddc9129a
```

The checkpoint contains the complete state. Instantiate `build_mobilenetv2` with 30 classes,
`pretrained=False`, mode `partial`, four unfrozen blocks, then load the state dictionary with
`strict=True`. Using `pretrained=False` at analysis time avoids an unnecessary download; the formal
training configuration still records ImageNet pretraining.

## Notebook safety

- Load saved CSV/JSON/PNG files for reported metrics; do not rerun final test inference.
- Do not use test results for new checkpoint selection, tuning or training.
- Do not mix validation metrics from different epochs.
- Configure raw images separately through `LEAFSNAP_DATA_ROOT` or `data_root` only when a Notebook
  genuinely needs example images.
- Do not regenerate metadata or splits.

`FILE_MANIFEST.csv` contains relative path, byte size and SHA-256 for every other file. It omits
itself because a file cannot contain a stable hash of its own final contents.
