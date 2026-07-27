Part 4 model reselection run

This directory isolates the new ResNet18 and EfficientNet-B0 training run from
all existing project outputs.

Run order:

1. configs/resnet18_frozen.yaml
2. configs/resnet18_layer4.yaml
3. configs/efficientnet_b0_frozen.yaml
4. configs/efficientnet_b0_finetune.yaml

All generated checkpoints and metrics are written below:

reselection_20260727_run1/outputs/

Protocol record:

- ResNet18 and EfficientNet-B0 were compared using validation metrics.
- EfficientNet-B0 was selected before final test evaluation.
- Selected validation accuracy: 0.9716242661448141.
- Final test accuracy: 0.978021978021978.
- Final test Macro-F1: 0.9794260041627031.
- The final test result must not be used for additional model selection or
  hyperparameter tuning.

Tracked reselection tables and figures are generated with:

python part4_plot_reselection_results.py
