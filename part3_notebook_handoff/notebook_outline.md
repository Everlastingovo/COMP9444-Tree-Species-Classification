# Suggested Part 3 Notebook outline

Keep the Notebook factual and load the copied CSV/JSON/PNG artifacts instead of rerunning formal
training or final test inference.

## 1. Part 3 objective

- State the 30-class Leafsnap classification task.
- Define the comparison: Frozen backbone versus Partial fine-tuning.
- State that the split and class mapping are locked Part 2 inputs.

## 2. MobileNetV2 motivation

- Briefly introduce ImageNet transfer learning and the lightweight MobileNetV2 architecture.
- Avoid adding performance claims not supported by the project results or cited literature.
- Reference `src/models/mobilenetv2.py`.

## 3. Dataset interface and preprocessing

- Show the 6,757-image split/source table from `data_summary.md`.
- Show the `(image, target)` interface and `[batch, 3, 224, 224]` shape.
- Explain ImageNet normalization and the train versus deterministic validation/test transforms.
- Reference `data/statistics/`, `src/data/dataset.py` and `src/data/transforms.py`.

## 4. Transfer-learning strategy

- Diagram or table: Frozen features versus Partial blocks 15-18.
- Include parameter counts and differential learning rates from `model_method_summary.md`.
- Explain frozen BatchNorm evaluation behaviour.

## 5. Frozen experiment

- Load `experiments/frozen/history.csv`.
- Display all four Frozen curves.
- State selected epoch 15 and its validation metrics.

## 6. Partial fine-tuning experiment

- State initialization from the locked Frozen checkpoint.
- Load `experiments/partial/history.csv`.
- Display all four Partial curves.
- State selected epoch 14 and its validation metrics.

## 7. Validation comparison

- Insert the validation columns from `notebook_results_tables.md` or `.csv`.
- Emphasize that every value comes from its Macro-F1-selected checkpoint epoch.
- Do not substitute Partial's epoch-13 minimum loss into the epoch-14 result row.

## 8. Final locked test evaluation

- State the protocol before displaying test numbers.
- Insert the test columns from the results table.
- Reference `final_test/evaluation_manifest.json` and `evaluation_protocol.md`.

## 9. Per-class and confusion analysis

- Insert both confusion-matrix PNGs.
- Use the classification-report CSVs for weakest-class tables.
- Discuss only the recorded confusion pairs in `error_analysis_summary.md`.

## 10. Lab versus field observations

- Insert the four-row source table.
- Note that Partial's observed improvement is mainly from field.
- State the 241-image/29-class field limitation.

## 11. Strengths, limitations and future work

- Strengths: fixed leakage-audited split, validation-only selection, multiple balanced metrics,
  reproducible checkpoints and one-time test protocol.
- Limitations: two transfer-learning regimes, one seed, class/source imbalance, small field subset,
  no controlled speed benchmark and no proof against visually near-duplicate leaves.
- Future work may mention XAI or additional controlled experiments, but must not imply they were run.

## 12. Reproducibility and contribution statement

- Summarize environment, seed, paths, checkpoint hashes and commands from
  `environment_and_reproducibility.md`.
- Insert or adapt `contribution_statement.md` without claiming other members' work.
