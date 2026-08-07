# Part 3: MobileNetV2 transfer learning

Part 3 reuses the locked Part 2 metadata and split CSV files. It must not regenerate them. The
entry point delegates data loading, the training loop, validation-based checkpoint selection and
final test loading to the shared trainer:

```powershell
python part3_mobilenetv2.py --data-root "D:\path\to\leafsnap-dataset-30subset"
```

The first pretrained run may need access to the torchvision ImageNet weights cache. The model can
also be built with `pretrained=False` for offline unit and smoke tests, but that is not the intended
formal transfer-learning experiment.

## Configurations

- `configs/mobilenetv2_frozen.yaml`: freezes every feature block and trains only the classifier at
  learning rate `0.001`.
- `configs/mobilenetv2_partial.yaml`: unfreezes the last four feature blocks and the classifier at
  learning rate `0.0001`.
- `full` remains supported by the model API, but no full-fine-tuning experiment is scheduled yet.

Both experiment configs use 224 x 224 inputs, ImageNet mean/std normalization, AdamW, seed 42 and
the checked-in `data/splits/{train,val,test}.csv` files. The existing Baseline continues to use its
default `(0.5, 0.5, 0.5)` normalization.

Select partial fine-tuning with:

```powershell
python part3_mobilenetv2.py `
  --config configs/mobilenetv2_partial.yaml `
  --data-root "D:\path\to\leafsnap-dataset-30subset"
```

## Lightweight verification

Run unit and data-interface checks without formal training:

```powershell
pytest -q tests/test_mobilenetv2.py tests/test_part2_data_integrity.py
```

For a one-batch end-to-end trainer smoke run, explicitly override the epoch and worker settings and
write only to a disposable output directory:

```powershell
python part3_mobilenetv2.py `
  --data-root "D:\path\to\leafsnap-dataset-30subset" `
  --epochs 1 --max-batches 1 --num-workers 0 `
  --output-dir outputs/mobilenetv2_smoke
```

Do not use test-set results for model selection. The shared trainer saves the checkpoint with the
best validation accuracy and evaluates the test split once after selection.
