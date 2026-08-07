# Notebook-ready results

All validation values below come from the single epoch selected by validation Macro-F1. They do
not combine the best individual metrics from different epochs. In particular, Partial's minimum
validation loss occurred at epoch 13, but the locked epoch-14 checkpoint has validation loss
0.175449 and is the value reported here.

| Model | Selected epoch | Val loss | Val Top-1 | Val Top-5 | Val Macro-F1 | Test loss | Test Top-1 | Test Top-5 | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 15 | 0.289849 | 93.0528% | 99.8043% | 93.1414% | 0.286923 | 92.6074% | 99.8002% | 93.1002% |
| Partial | 14 | 0.175449 | 94.9119% | 99.9022% | 95.0317% | 0.189515 | 93.8062% | 100.0000% | 93.9869% |
| Partial - Frozen | - | -0.114400 | +1.8591 pp | +0.0978 pp | +1.8903 pp | -0.097408 | +1.1988 pp | +0.1998 pp | +0.8867 pp |

Recommended Notebook insertions:

- Use this table for the primary validation/test comparison.
- Insert `experiments/*/curves/` for training dynamics.
- Insert `final_test/*_confusion_matrix.png` after the locked test table.
- Use the per-class report CSVs for a compact weakest-class table.
