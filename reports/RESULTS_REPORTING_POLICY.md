# Part 5 Results Reporting Policy

This document defines which result must be used in the report, notebook, and
presentation. It prevents owner-reported locked results from being mixed with
local post-hoc reruns.

## Headline results

Use these values whenever stating the final test accuracy of a model:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Top-5 |
|---|---:|---:|---:|---:|
| Baseline Custom CNN | 80.52% | 80.84% | 80.09% | 98.60% |
| MobileNetV2 Partial | 93.81% | 93.99% | 93.80% | 100.00% |
| ResNet18 Layer4 | 97.10% | 97.22% | 97.11% | 99.80% |

The headline accuracy comes from each model owner's validation-selected
checkpoint and locked final-test result. Macro-F1, Weighted-F1, and Top-5 are
calculated from the traceable detailed prediction file. The Baseline detailed
file is a local post-hoc rerun.

## Detailed analysis

Use the standardized prediction files for per-class metrics, lab/field metrics,
confusion matrices, error analysis, and Grad-CAM sample selection.

| Model | Detailed prediction accuracy | Difference from headline |
|---|---:|---:|
| Baseline Custom CNN | 80.42% | -0.10 percentage points |
| MobileNetV2 Partial | 93.81% | 0.00 percentage points |
| ResNet18 Layer4 | 97.10% | 0.00 percentage points |

The Baseline local rerun changes one prediction relative to the owner's
summary. Report the owner value for headline accuracy and disclose the local
80.42% only when describing detailed post-hoc analysis.

## Brightness experiment

The brightness experiment is a separate local CPU rerun. Every change must be
calculated relative to the same run's brightness-1.0 result:

| Model | Brightness-1.0 baseline | Maximum accuracy drop |
|---|---:|---:|
| Baseline Custom CNN | 80.42% | 30.37 percentage points |
| MobileNetV2 Partial | 93.81% | 2.80 percentage points |
| ResNet18 Layer4 | 97.00% | 2.50 percentage points |

Do not replace the 97.10% headline ResNet18 result with the local 97.00%
brightness baseline. The one-image difference is a cross-platform inference
difference, not evidence of model degradation.

## Source comparison

Use the following values for the fixed 760-image lab and 241-image field
subsets:

| Model | Lab accuracy | Field accuracy | Lab minus field |
|---|---:|---:|---:|
| Baseline Custom CNN | 82.63% | 73.44% | 9.19 points |
| MobileNetV2 Partial | 93.03% | 96.27% | -3.24 points |
| ResNet18 Layer4 | 97.50% | 95.85% | 1.65 points |

The field subset has only 241 images and represents 29 classes. MobileNetV2's
higher field accuracy must therefore be presented as an observed subset result,
not proof that field images are generally easier.

## Efficiency experiment

Latency values apply only to the recorded Apple CPU, Torch 2.10.0, eight Torch
threads, and model-only forward passes. They must not be generalized to CUDA,
mobile accelerators, or optimized deployment runtimes.

## Formatting rules

- Round percentages to two decimal places in prose and presentation slides.
- Describe differences as percentage points, not percentages.
- Label brightness and latency results as local post-hoc experiments.
- State that checkpoints were frozen and test results were not used for tuning.
- Keep checkpoint hashes and full-precision CSV values in the reproducibility
  material rather than the main report narrative.
