# Part 4 Extension: EfficientNet-B0

## Protocol status

EfficientNet-B0 is a post-hoc, validation-only lightweight extension. It was added after the ResNet18 test result had already been observed. To preserve evaluation integrity, EfficientNet-B0 was trained and compared only on the corrected training and validation splits and was never evaluated on the test split.

## Model and setup

The implementation uses `EfficientNet_B0_Weights.DEFAULT`, replaces the 1,000-class classifier with a 30-class linear layer, and applies ImageNet normalisation to 224 x 224 RGB inputs.

| Stage | Trainable scope | Trainable parameters | Learning rate | Epochs |
|---|---|---:|---:|---:|
| Frozen head | Classifier | 38,430 | 1e-3 | 5 |
| Fine-tuning | Final two feature blocks + classifier | 1,167,822 | 3e-5 | 10 |

The adapted network has 4,045,978 total parameters. The fine-tuning stage starts from the best frozen-head checkpoint. Both stages use seed 42, batch size 16, weight decay `1e-4`, identical corrected splits, and the shared augmentation pipeline.

## Validation results

| Experiment | Best epoch | Validation accuracy | Validation Macro-F1 | Validation Weighted-F1 | Top-5 accuracy |
|---|---:|---:|---:|---:|---:|
| Frozen head | 5 | 94.91% | Not calculated | Not calculated | Not calculated |
| Fine-tuned | 9 | **97.16%** | **97.08%** | **97.15%** | **99.90%** |

The fine-tuned validation accuracy is 0.10 percentage points above the selected ResNet18 validation accuracy. This small single-split difference is descriptive and is not evidence of a statistically significant improvement. ResNet18 remains the locked final model because it was selected under the original test protocol and has the only valid final test evaluation.

## Reproducibility

```bash
python part4_efficientnet_b0.py --config configs/efficientnet_b0_frozen.yaml
python part4_efficientnet_b0.py --config configs/efficientnet_b0_finetune.yaml
python part4_evaluate_efficientnet_b0.py
```

The evaluator is structurally validation-only: it provides no split argument and reads only `data/splits/val.csv`.

## Deliverables

- `src/models/efficientnet_b0.py`
- `part4_efficientnet_b0.py`
- `part4_evaluate_efficientnet_b0.py`
- `configs/efficientnet_b0_frozen.yaml`
- `configs/efficientnet_b0_finetune.yaml`
- `report/tables/efficientnet_b0_experiments.csv`
- `report/figures/efficientnet_b0/efficientnet_b0_training_curves.png`
- `report/figures/efficientnet_b0/transfer_model_validation_comparison.png`

Checkpoints and detailed validation predictions remain under `outputs/efficientnet_b0/` and are ignored by Git.
