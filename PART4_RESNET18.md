# Part 4: ResNet18 Transfer Learning

## Responsibility and scope

Part 4 implements and evaluates a ResNet18 transfer-learning model for the 30-class Leafsnap tree-species classification task. The implementation uses the TorchVision architecture and ImageNet weights, then adds project-specific classifier replacement, staged fine-tuning, frozen BatchNorm handling, fixed-split integration, checkpoint transfer, multi-class evaluation, and reproducible reporting.

All results in this document were regenerated after Part 2 corrected exact-content leakage, restored omitted laboratory images, and relocked the dataset splits. Results from the superseded 7,239-image split are not used.

## Corrected data and preprocessing

| Split | Images |
|---|---:|
| Training | 4,734 |
| Validation | 1,022 |
| Test | 1,001 |
| **Total** | **6,757** |

The corrected split contains 30 classes and has no path or SHA-256 content overlap between train, validation, and test. Part 4 reused the checked-in split files without regeneration.

Inputs are 224 x 224 RGB images. Training uses the shared augmentation pipeline; validation and test use deterministic preprocessing. ImageNet normalisation is applied:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## Model implementation

The model is created with `torchvision.models.ResNet18_Weights.DEFAULT`. The original 1,000-class fully connected layer is replaced by a 30-class layer.

The implementation supports three trainable scopes:

- `head`: train only the final classifier.
- `layer4`: train the final residual block and classifier.
- `all`: fine-tune the complete network.

The adapted model contains 11,191,902 parameters. The frozen-head stage has 15,390 trainable parameters; the Layer4 stage has 8,409,118. Frozen BatchNorm modules remain in evaluation mode during training so their running statistics do not change unintentionally.

## Controlled experiments

All experiments use seed 42, batch size 16, weight decay `1e-4`, 224 x 224 inputs, identical augmentation, ImageNet normalisation, and the same corrected splits. Only the trainable scope or Layer4 learning rate changes.

| Experiment | Scope | Learning rate | Epochs | Best epoch | Validation accuracy | Validation Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Frozen head | Head only | 1e-3 | 5 | 4 | 89.73% | 89.67% |
| Layer4 | Layer4 + head | 1e-4 | 10 | 10 | 96.77% | 96.69% |
| Layer4 low LR | Layer4 + head | 3e-5 | 10 | 9 | **97.06%** | **97.03%** |

Both Layer4 runs start from the best corrected-data frozen checkpoint. The `3e-5` checkpoint was selected using validation results before final test evaluation.

## Locked final test result

Selected checkpoint:

```text
outputs/resnet18/layer4_lr3e5/best_model.pt
```

The checkpoint was evaluated once on the corrected 1,001-image test split. No model or hyperparameter selection occurred after observing the result.

| Metric | Score |
|---|---:|
| Accuracy | **97.10%** |
| Macro Precision | 97.31% |
| Macro Recall | 97.24% |
| Macro-F1 | **97.22%** |
| Weighted Precision | 97.22% |
| Weighted Recall | 97.10% |
| Weighted-F1 | **97.11%** |
| Top-5 Accuracy | 99.80% |

Test accuracy is 0.04 percentage points above validation accuracy, and test Macro-F1 is 0.20 points above validation Macro-F1. The close results indicate stable held-out performance.

## Per-class observations

The five lowest test F1 scores are:

| Species | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `ulmus_americana` | 87.10% | 84.38% | 85.71% | 32 |
| `diospyros_virginiana` | 88.10% | 97.37% | 92.50% | 38 |
| `ostrya_virginiana` | 86.11% | 100.00% | 92.54% | 31 |
| `styrax_japonica` | 91.89% | 94.44% | 93.15% | 36 |
| `ulmus_rubra` | 97.78% | 89.80% | 93.62% | 49 |

`ulmus_americana` is the weakest class and has both imperfect precision and recall, suggesting confusion in both directions. `ostrya_virginiana` has perfect recall but lower precision, so other species are sometimes assigned to it. These classes are suitable priorities for confusion-matrix inspection and explainability work.

## Reproducibility

```bash
python part4_resnet18.py --config configs/resnet18_frozen.yaml
python part4_resnet18.py --config configs/resnet18_layer4.yaml
python part4_resnet18.py --config configs/resnet18_layer4_lr3e5.yaml
```

Validation evaluation is the default:

```bash
python part4_evaluate_resnet18.py
```

The final test command is recorded only for provenance and must not be rerun for model selection:

```bash
python part4_evaluate_resnet18.py --split test
```

Saved results can be converted into tracked tables and figures without model or dataset evaluation:

```bash
python part4_plot_results.py
```

## Deliverables

- `src/models/resnet18.py`
- `part4_resnet18.py`
- `part4_evaluate_resnet18.py`
- `part4_plot_results.py`
- `configs/resnet18_frozen.yaml`
- `configs/resnet18_layer4.yaml`
- `configs/resnet18_layer4_lr3e5.yaml`
- `report/tables/resnet18_experiments.csv`
- `report/tables/resnet18_final_test_metrics.csv`
- `report/tables/resnet18_test_per_class_metrics.csv`
- `report/figures/resnet18/resnet18_training_curves.png`
- `report/figures/resnet18/resnet18_validation_comparison.png`
- `report/figures/resnet18/resnet18_test_per_class_f1.png`

Checkpoints and detailed predictions remain under `outputs/resnet18/` and are ignored by Git.

## Limitations and handoff

- Laboratory and field performance has not been reported separately.
- Only a small controlled hyperparameter set was explored.
- Runtime was not captured consistently and should not be compared retrospectively.
- The final group analysis can add source-level metrics, confusion examples, and Grad-CAM visualisations without using the locked test set for additional tuning.

This material must be integrated into the group's single final Project Notebook rather than submitted as a separate notebook.
