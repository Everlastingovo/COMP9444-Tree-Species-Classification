# Error-analysis summary

## Weakest classes by test F1

Frozen:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `ostrya_virginiana` | 72.22% | 83.87% | 77.61% | 31 |
| `diospyros_virginiana` | 75.61% | 81.58% | 78.48% | 38 |
| `ulmus_rubra` | 80.85% | 77.55% | 79.17% | 49 |
| `styrax_japonica` | 75.61% | 86.11% | 80.52% | 36 |
| `ulmus_americana` | 96.00% | 75.00% | 84.21% | 32 |

Partial:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `diospyros_virginiana` | 77.50% | 81.58% | 79.49% | 38 |
| `ulmus_americana` | 95.65% | 68.75% | 80.00% | 32 |
| `ulmus_rubra` | 86.67% | 79.59% | 82.98% | 49 |
| `ulmus_pumila` | 75.68% | 93.33% | 83.58% | 30 |
| `ostrya_virginiana` | 80.56% | 93.55% | 86.57% | 31 |

## Main confusion pairs

Frozen's largest unordered pairs were:

- `diospyros_virginiana` / `styrax_japonica`: 9 errors (5 one direction, 4 reverse).
- `ostrya_virginiana` / `ulmus_rubra`: 9 errors (3 one direction, 6 reverse).
- `ulmus_pumila` / `ulmus_rubra`: 7 errors (2 one direction, 5 reverse).

Partial's largest unordered pairs were:

- `ulmus_pumila` / `ulmus_rubra`: 9 errors (1 one direction, 8 reverse).
- `diospyros_virginiana` / `styrax_japonica`: 5 errors (2 one direction, 3 reverse).
- `ostrya_virginiana` / `ulmus_americana`: 5 errors (1 one direction, 4 reverse).

## Lab versus field

| Model/source | Images | Top-1 | Macro-F1 |
|---|---:|---:|---:|
| Frozen lab | 760 | 93.0263% | 93.4881% |
| Frozen field | 241 | 91.2863% | 90.0588% |
| Partial lab | 760 | 93.0263% | 93.0071% |
| Partial field | 241 | 96.2656% | 96.4447% |

The overall Partial improvement is mainly associated with the field subset: field Top-1 increases
by 4.9793 percentage points and field Macro-F1 by 6.3859 points. Lab Top-1 is unchanged, while lab
Macro-F1 decreases by 0.4810 points.

This is a fixed descriptive analysis, not evidence for new tuning. Field contains only 241 images
and covers 29 classes; `quercus_muehlenbergii` has no field test support. The smaller subset makes
its percentages more variable than the overall result.

Frozen ran first and therefore may include CUDA/DataLoader warm-up overhead. Its 19.973-second
inference time and Partial's 14.772 seconds should not be used as a controlled model-speed
comparison. The architectures have the same inference graph regardless of which training blocks
were unfrozen.
