# Locked evaluation protocol

1. Frozen and Partial checkpoints were selected before any test evaluation.
2. Selection used validation Macro-F1 only.
3. Frozen locked epoch: 15; validation Macro-F1: 93.1414%.
4. Partial locked epoch: 14; validation Macro-F1: 95.0317%.
5. Each locked checkpoint was evaluated on the 1,001-image test split exactly once.
6. The evaluation used deterministic `224 x 224` ImageNet preprocessing and `shuffle=False`.
7. No TTA, ensemble or random crop was used.
8. Test metrics were not used to select a checkpoint, choose an epoch or tune hyperparameters.
9. No training occurred after test evaluation.
10. This Notebook handoff does not rerun test inference.

Checkpoint identities:

```text
Frozen SHA-256
6c708cc6713f02732da137e9ac4396a939696c285b125b46df85213c8c421fd3

Partial SHA-256
9716e5f311abb5c828539f9ab7963c7444f6caf2cbc004e3c4c91838ddc9129a
```

The machine-readable source of record is `final_test/evaluation_manifest.json`, which records
`test_evaluation_count: 1` for each checkpoint and sets `no_tta`, `no_ensemble`,
`no_random_crop`, `no_training_after_test` and `checkpoint_selection_used_test` appropriately.
