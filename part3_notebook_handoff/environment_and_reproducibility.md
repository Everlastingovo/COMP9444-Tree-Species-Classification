# Environment and reproducibility

## Repository snapshot

- Project root: `D:\Desktop\9444\COMP9444-Tree-Species-Classification`
- Branch: `part3`
- HEAD: `9cfe873472752a66e7906a86a5a1958b106a7d8e`
- Commit: `9cfe873 Complete MobileNetV2 fine-tuning and final evaluation`
- Random seed: `42`

## Python and GPU environment

- Python executable: `D:\Desktop\9444\COMP9444-Tree-Species-Classification\.venv\Scripts\python.exe`
- Python: `3.12.10`
- PyTorch: `2.13.0+cu130`
- torchvision: `0.28.0+cu130`
- CUDA runtime reported by PyTorch: `13.0`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- PyYAML: `6.0.3`
- pytest: `9.1.1`

## Data root

The formal experiments used:

```text
D:\Desktop\9444\leafsnap-dataset-30subset 1\leafsnap-dataset-30subset
```

Split paths are relative to the Leafsnap subset root and begin with `dataset/`. Configure another
machine either with `--data-root` or:

```powershell
$env:LEAFSNAP_DATA_ROOT = "D:\path\to\leafsnap-dataset-30subset"
```

Do not regenerate metadata or splits. The checked-in, content-cleaned split is the experiment
contract.

## Formal command records

These commands document the completed runs. The final-test command is historical and must not be
run again merely to rebuild a Notebook.

Frozen backbone:

```powershell
.\.venv\Scripts\python.exe part3_mobilenetv2.py `
  --config configs\mobilenetv2_frozen.yaml `
  --data-root "D:\Desktop\9444\leafsnap-dataset-30subset 1\leafsnap-dataset-30subset"
```

Partial fine-tuning:

```powershell
.\.venv\Scripts\python.exe part3_mobilenetv2.py `
  --config configs\mobilenetv2_partial.yaml `
  --data-root "D:\Desktop\9444\leafsnap-dataset-30subset 1\leafsnap-dataset-30subset"
```

One-time locked final test evaluation:

```powershell
.\.venv\Scripts\python.exe part3_final_test.py `
  --data-root "D:\Desktop\9444\leafsnap-dataset-30subset 1\leafsnap-dataset-30subset"
```

## Recorded verification results

The completed Part 3 verification recorded:

- MobileNetV2, Trainer and final-evaluation tests: **14 passed in 12.03 s**.
- Part 2 data-integrity tests: **5 passed in 47.96 s**.
- Combined recorded test cases: **19 passed**.

Commands used for those checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests\test_final_test_evaluation.py tests\test_mobilenetv2.py tests\test_training_metrics.py

$env:LEAFSNAP_DATA_ROOT = "D:\Desktop\9444\leafsnap-dataset-30subset 1\leafsnap-dataset-30subset"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_part2_data_integrity.py
```

The packaging task did not rerun training, model test inference, metadata generation or split
generation.
