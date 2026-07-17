# Part 4: ResNet18 Transfer Learning

## Responsibility and scope

Part 4 implements and evaluates a ResNet18 transfer-learning model for the 30-class Leafsnap tree-species classification task. The work covers model construction, staged fine-tuning, controlled learning-rate comparison, checkpoint selection, validation and final test evaluation, result visualisation, and handoff materials.

The implementation builds on the ResNet18 architecture and ImageNet weights provided by TorchVision. The project-specific work includes replacing the classifier, controlling trainable layers, preserving frozen BatchNorm statistics, integrating the fixed Part 2 data splits, applying ImageNet normalisation, loading checkpoints between training stages, calculating multi-class metrics, and generating reproducible result tables and figures.

## Data and preprocessing

All experiments use the fixed Part 2 split files and class mapping. No split was regenerated during Part 4.

| Split | Images |
|---|---:|
| Training | 5,070 |
| Validation | 1,094 |
| Test | 1,075 |
| Total | 7,239 |

The model input is 224 x 224 RGB. Training images use the shared augmentation pipeline. Validation and test images use deterministic resizing and padding without random augmentation. ResNet18 uses ImageNet normalisation:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The original standard normalisation remains available as the default for the custom CNN baseline, so the transfer-learning change does not silently alter existing baseline behaviour.

## Model implementation

The model is created with `torchvision.models.ResNet18_Weights.DEFAULT`. Its original 1,000-class fully connected layer is replaced with a 30-class layer.

The implementation supports three trainable scopes:

- `head`: train only the final fully connected layer.
- `layer4`: train the final residual block and classifier.
- `all`: fine-tune the entire network.

The 30-class model contains 11,191,902 total parameters. The frozen-head stage has 15,390 trainable parameters, while the Layer4 stage has 8,409,118 trainable parameters. Frozen BatchNorm modules remain in evaluation mode during training so that their running statistics are not updated unintentionally.

## Experimental design

All experiments use the same split, class mapping, seed, input size, augmentation, normalisation, batch size, and weight decay. The controlled comparison changes only the trainable scope or Layer4 learning rate.

| Experiment | Scope | Learning rate | Epochs | Best epoch | Validation accuracy | Validation Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Frozen head | Head only | 1e-3 | 5 | 4 | 89.49% | 89.26% |
| Layer4 | Layer4 + head | 1e-4 | 10 | 7 | 95.98% | 95.90% |
| Layer4 low LR | Layer4 + head | 3e-5 | 10 | 10 | **96.44%** | **96.08%** |

The Layer4 stages start from the best frozen-head checkpoint. Model selection uses validation performance only. The test set was evaluated once after the `3e-5` model was selected as the final checkpoint.

## Final test results

The selected checkpoint is:

```text
outputs/resnet18/layer4_lr3e5/best_model.pt
```

Final test metrics on 1,075 images are:

| Metric | Score |
|---|---:|
| Accuracy | 95.53% |
| Macro Precision | 96.47% |
| Macro Recall | 95.29% |
| Macro-F1 | 95.39% |
| Weighted Precision | 96.22% |
| Weighted Recall | 95.53% |
| Weighted-F1 | 95.39% |
| Top-5 Accuracy | 100.00% |

The validation-to-test accuracy gap is 0.91 percentage points, and the Macro-F1 gap is 0.69 percentage points. These small gaps indicate that the selected model generalises well to the held-out test set.

## Per-class observations

Most species achieve high test F1 scores, but five classes are notably weaker:

| Species | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `quercus_muehlenbergii` | 100.00% | 47.06% | 64.00% | 34 |
| `prunus_sargentii` | 67.74% | 100.00% | 80.77% | 42 |
| `ostrya_virginiana` | 82.86% | 87.88% | 85.29% | 33 |
| `ulmus_americana` | 92.86% | 81.25% | 86.67% | 32 |
| `catalpa_bignonioides` | 100.00% | 81.25% | 89.66% | 32 |

`quercus_muehlenbergii` is the main failure class. Its perfect precision but low recall means predictions assigned to this class are reliable, but many true examples are classified as other species. `prunus_sargentii` shows the opposite pattern: perfect recall but lower precision, indicating that examples from other classes are sometimes assigned to it. These classes should receive priority in the group's confusion-matrix and error-analysis work.

## Reproducible commands

Frozen-head training:

```bash
python part4_resnet18.py --config configs/resnet18_frozen.yaml
```

Layer4 training with learning rate `1e-4`:

```bash
python part4_resnet18.py --config configs/resnet18_layer4.yaml
```

Layer4 training with learning rate `3e-5`:

```bash
python part4_resnet18.py --config configs/resnet18_layer4_lr3e5.yaml
```

Validation evaluation is the default:

```bash
python part4_evaluate_resnet18.py
```

The final test command is recorded for reproducibility but must not be rerun for further model selection:

```bash
python part4_evaluate_resnet18.py --split test
```

Generate tables and figures from saved results without loading a dataset or running a model:

```bash
python part4_plot_results.py
```

## Deliverables

### Code and configuration

- `src/models/resnet18.py`: model construction and trainable-scope control.
- `src/data/transforms.py`: optional ImageNet normalisation.
- `src/data/dataset.py`: normalisation selection through the dataset interface.
- `src/training/trainer.py`: frozen BatchNorm handling.
- `src/utils/device.py`: CUDA, MPS, and CPU device selection.
- `part4_resnet18.py`: staged training entry point.
- `part4_evaluate_resnet18.py`: validation and explicit final-test evaluation.
- `part4_plot_results.py`: result table and figure generation.
- `configs/resnet18_frozen.yaml`: frozen-head experiment.
- `configs/resnet18_layer4.yaml`: Layer4 experiment with learning rate `1e-4`.
- `configs/resnet18_layer4_lr3e5.yaml`: Layer4 experiment with learning rate `3e-5`.

### Tracked result tables

- `report/tables/resnet18_experiments.csv`
- `report/tables/resnet18_final_test_metrics.csv`
- `report/tables/resnet18_test_per_class_metrics.csv`

### Tracked figures

- `report/figures/resnet18/resnet18_training_curves.png`
- `report/figures/resnet18/resnet18_validation_comparison.png`
- `report/figures/resnet18/resnet18_test_per_class_f1.png`

Checkpoints and detailed local predictions remain under `outputs/resnet18/` and are ignored by Git because they are large or machine-specific artifacts.

## Limitations and future work

- The model still struggles with a small number of visually similar species, especially `quercus_muehlenbergii`.
- The dataset contains both controlled laboratory images and real-world field images. A separate source-level evaluation is needed to quantify the domain gap.
- The selected model was tuned on a small, controlled set of configurations rather than an exhaustive hyperparameter search.
- The final group analysis should include confusion matrices, representative error cases, and Grad-CAM visualisations.
- Training-time measurements were not captured consistently across all experiments and should not be compared without rerunning under a controlled timing protocol.

## Handoff notes

The final model was selected using validation Accuracy and Macro-F1. The test set was then evaluated once and was not used for further tuning. Member 5 can use the tracked per-class table and local prediction file for confusion analysis, error inspection, and explainability work. The ResNet18 material should be merged into the group's single final Project Notebook rather than submitted as a separate notebook.
