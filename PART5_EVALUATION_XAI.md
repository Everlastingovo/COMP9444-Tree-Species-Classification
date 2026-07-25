# Part 5: Unified Evaluation, Error Analysis, and Grad-CAM

## Scope

Part 5 evaluates the locked Baseline, MobileNetV2, and ResNet18 models. It does
not retrain or tune the models. All models must use the same corrected test split,
which contains 1,001 images from 30 classes.

The main goals are:

- calculate the same metrics for every model;
- compare model accuracy, Macro-F1, Weighted-F1, Top-5 accuracy, and efficiency;
- analyse per-class performance and common confusion pairs;
- compare laboratory and field performance;
- inspect high-confidence errors and low-confidence correct predictions;
- generate Grad-CAM examples for correct and incorrect predictions;
- prepare figures and conclusions for the notebook, report, and presentation.

## Completed foundation

The first implementation stage includes:

- `src/evaluation/metrics.py`
  - Accuracy and Top-5 accuracy
  - Macro and weighted Precision, Recall, and F1
  - per-class metrics
  - lab and field metric breakdown
- `src/evaluation/confusion.py`
  - raw and normalized confusion matrices
  - strongest confusion-pair detection
  - CSV and PNG output
- `src/evaluation/error_analysis.py`
  - traceable per-image prediction rows
  - high-confidence error selection
  - low-confidence correct selection
  - confusion-pair filtering
  - error-case image grids
- `src/evaluation/gradcam.py`
  - automatic final convolution-layer selection
  - Grad-CAM generation for frozen and fine-tuned models
  - heatmap overlays and report-ready image grids
- `notebooks/06_evaluation_xai.ipynb`
  - marking-criteria-aligned section structure

Run the current tests with:

```bash
python -m pytest -q tests/test_part5_evaluation.py
```

## Required model handoff

Each model owner must provide:

1. the validation-selected checkpoint;
2. the exact configuration used for that checkpoint;
3. the fixed `class_to_idx.json`;
4. overall and per-class metrics;
5. per-image predictions;
6. checkpoint SHA-256 when available.

The preferred per-image prediction columns are:

```text
path
source
true_idx
true_label
predicted_idx
predicted_label
confidence
correct
top5_labels
```

Expected selected checkpoints:

```text
outputs/baseline/.../best_model.pt
outputs/mobilenetv2/partial_seed42/best_mobilenetv2_partial.pt
outputs/resnet18/layer4_lr3e5/best_model.pt
```

The Baseline must be rerun on the corrected 6,757-image dataset. The old
field-only result with 314 test images is not directly comparable.

## Planned final outputs

The completed evaluation will save:

```text
outputs/comparison/model_comparison.csv
outputs/comparison/model_comparison.png
outputs/comparison/per_class_metrics.csv
outputs/comparison/confusion_matrix_raw.png
outputs/comparison/confusion_matrix_normalized.png
outputs/comparison/source_comparison.csv
outputs/comparison/source_comparison.png
outputs/comparison/error_cases.csv
outputs/comparison/error_cases.png
outputs/comparison/gradcam_examples.png
```

The locked Baseline, MobileNetV2, and ResNet18 handoffs and raw images are now
available. Unified model comparison, error analysis, and Grad-CAM outputs have
been generated under `outputs/comparison/`.

## Current handoff status

### Baseline

The refreshed Baseline handoff has been validated against the locked project
metadata:

- checkpoint SHA-256:
  `2ae3f85a28d46e2207c5cf6a8c37efe817af5794c8c2ea0845a6ebb02206367d`;
- the 30-class mapping matches exactly;
- all 4,734 train, 1,022 validation, and 1,001 test paths and labels match;
- the test split contains 760 lab and 241 field images;
- the checkpoint loads strictly into `CustomCNN`;
- the Grad-CAM target is `model.features[3][3]`.

The handoff's `summary.json` reports 80.52% test accuracy. Post-hoc inference
on macOS with PyTorch 2.10.0 and Pillow 10.4.0 gives 80.42%, a one-image
difference. The evaluation manifest records both results and their difference.
Use the original handoff result as the owner's official overall accuracy and
the reproducible post-hoc predictions for detailed error, source, and Grad-CAM
analysis. This small platform difference must be disclosed if those values are
presented together.

Generate the Baseline prediction and analysis outputs with:

```bash
python part5_export_baseline_predictions.py \
  --checkpoint outputs/baseline/member5_handoff/best_custom_cnn.pt \
  --data-root /path/to/leafsnap-dataset-30subset \
  --reported-summary outputs/baseline/member5_handoff/summary.json \
  --output-dir outputs/baseline/member5_handoff

python part5_analyse_predictions.py \
  --predictions outputs/baseline/member5_handoff/per_image_predictions.csv \
  --output-dir outputs/comparison/baseline \
  --model-name Baseline-CustomCNN \
  --data-root /path/to/leafsnap-dataset-30subset

python part5_generate_baseline_gradcam.py \
  --checkpoint outputs/baseline/member5_handoff/best_custom_cnn.pt \
  --predictions outputs/baseline/member5_handoff/per_image_predictions.csv \
  --data-root /path/to/leafsnap-dataset-30subset \
  --output-dir outputs/comparison/baseline
```

### ResNet18

The ResNet18 handoff passed all integrity and semantic checks:

- all 13 files covered by `SHA256SUMS.txt` match;
- checkpoint SHA-256:
  `9ece5f02e59de428a475f5fe864cc4d4f97ec5dad3296fdb4d95244728aa0903`;
- the checkpoint and metadata use the locked 30-class mapping;
- all 1,001 test rows match the locked split in the same order;
- all prediction paths are unique and contain valid Top-5 results;
- unified metric recomputation matches the handoff results.

The final ResNet18 result is 97.10% accuracy, 97.22% Macro-F1, 97.11%
Weighted-F1, and 99.80% Top-5 accuracy. The recommended Grad-CAM target is
`model.layer4[-1]`.

Generate the transfer-model Grad-CAM outputs with:

```bash
python part5_generate_transfer_gradcam.py \
  --architecture mobilenetv2 \
  --checkpoint outputs/mobilenetv2/member5_handoff/checkpoint/best_mobilenetv2_partial.pt \
  --predictions outputs/comparison/mobilenetv2/predictions_standardized.csv \
  --data-root /path/to/leafsnap-dataset-30subset \
  --output-dir outputs/comparison/mobilenetv2

python part5_generate_transfer_gradcam.py \
  --architecture resnet18 \
  --checkpoint outputs/resnet18/member5_handoff/checkpoint/best_model.pt \
  --predictions outputs/comparison/resnet18/predictions_standardized.csv \
  --data-root /path/to/leafsnap-dataset-30subset \
  --output-dir outputs/comparison/resnet18
```

### Final comparison

Build the combined CSV files and report-ready figures with:

```bash
python part5_build_model_comparison.py \
  --reported-baseline-summary outputs/baseline/member5_handoff/summary.json
```

On the locked test split, ResNet18 is the strongest model, followed by the
partially fine-tuned MobileNetV2 and the custom CNN Baseline. ResNet18 reaches
95.85% field accuracy and has a 1.65 percentage-point lab-to-field gap. The
Baseline has the largest degradation, falling from 82.63% lab accuracy to
73.44% field accuracy. MobileNetV2 records 96.27% field accuracy, 3.24 points
higher than its lab accuracy; this should be interpreted alongside the smaller
field subset and its 29 represented classes.

## Efficiency experiment

Run the shared CPU benchmark with:

```bash
python part5_benchmark_efficiency.py
```

The benchmark uses the same Apple CPU, eight Torch threads, ten warm-up
iterations, and 50 measured batch-one iterations for all models. It measures
the model forward pass only and excludes file I/O and preprocessing. Each model
retains its locked deployment input size.

| Model | Parameters | Checkpoint | Median batch-one latency |
|---|---:|---:|---:|
| Baseline-CustomCNN | 1.21M | 4.87 MB | 7.73 ms |
| MobileNetV2-Partial | 2.26M | 9.30 MB | 57.85 ms |
| ResNet18-Layer4 | 11.19M | 44.85 MB | 17.52 ms |

The Baseline is smallest and fastest but substantially less accurate.
MobileNetV2 is much smaller than ResNet18, but its depthwise operations are
slower on this CPU backend. This latency ranking is hardware-specific and must
not be generalized to CUDA GPUs, mobile accelerators, or optimized deployment
runtimes.

## Brightness robustness experiment

Run the frozen-checkpoint stress test with:

```bash
python part5_brightness_robustness.py \
  --data-root /path/to/leafsnap-dataset-30subset
```

The test applies global RGB brightness factors `0.6`, `0.8`, `1.0`, `1.2`,
and `1.4` before each model's locked resize and normalization. The full locked
1,001-image test split is used at every factor. Checkpoints remain frozen, and
the results are not used for model selection.

| Model | Local clean accuracy | Mean perturbed accuracy | Worst factor | Maximum drop |
|---|---:|---:|---:|---:|
| Baseline-CustomCNN | 80.42% | 66.66% | 0.6 | 30.37 points |
| MobileNetV2-Partial | 93.81% | 92.28% | 1.4 | 2.80 points |
| ResNet18-Layer4 | 97.00% | 95.43% | 1.4 | 2.50 points |

At factor `0.6`, accuracy is 50.05% for the Baseline, 92.01% for MobileNetV2,
and 94.61% for ResNet18. The transfer models are therefore much more robust to
this controlled brightness shift. ResNet18's local clean recomputation differs
from the owner-reported result by one image (97.00% versus 97.10%), so all
brightness changes are calculated relative to the local 1.0 run.

Global brightness is only a controlled proxy for illumination. It does not
model spatial shadows, glare, color temperature, or sensor noise, so the
lab/field comparison remains the stronger real-world domain-shift analysis.

## Report packaging and result policy

Final tracked figures and tables are stored under `reports/figures/` and
`reports/tables/`. Use `reports/README.md` for the recommended main-report
figure order.

Before transferring any value into the report or presentation, check
`reports/RESULTS_REPORTING_POLICY.md`. In particular:

- use owner-reported locked accuracy for the three headline model results;
- use standardized prediction files for detailed class, source, and error
  analysis;
- use each local brightness-1.0 run only as the reference for brightness
  changes;
- describe all accuracy differences in percentage points;
- label CPU efficiency and brightness results as local post-hoc experiments.
