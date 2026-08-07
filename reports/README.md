# Part 5 Report Assets

All files in this directory are tracked copies of final outputs. Large
checkpoints, raw images, and intermediate predictions remain ignored under
`outputs/`.

## Recommended main-report figures

1. `figures/model_comparison.png`
   - Overall Accuracy, Macro-F1, Weighted-F1, and Top-5 comparison.
2. `figures/source_comparison.png`
   - Lab and field accuracy for all three models.
3. `figures/brightness_robustness.png`
   - Accuracy under five global brightness factors.
4. `figures/model_efficiency.png`
   - Parameter count and CPU batch-one latency.
5. `figures/resnet18_confusion_matrix_normalized.png`
   - Normalized final-model confusion matrix.
6. `figures/resnet18_error_cases.png`
   - Representative high-confidence ResNet18 errors.
7. `figures/resnet18_gradcam_examples.png`
   - Correct and incorrect ResNet18 Grad-CAM examples.

`figures/brightness_source_robustness.png` is useful when space allows. The
Baseline and MobileNetV2 confusion, error, and Grad-CAM figures are retained as
supplementary material.

## Tables

- `tables/model_comparison.csv`
- `tables/source_comparison.csv`
- `tables/model_efficiency.csv`
- `tables/brightness_robustness.csv`
- `tables/brightness_robustness_summary.csv`
- `tables/per_class_metrics.csv`
- `tables/resnet18_per_class_metrics.csv`
- `tables/resnet18_strongest_confusion_pairs.csv`
- `tables/result_reporting_policy.csv`

Use `RESULTS_REPORTING_POLICY.md` before transferring numbers into the report
or presentation.
