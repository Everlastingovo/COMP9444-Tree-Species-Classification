# MobileNetV2 model and training method

## Architecture

Part 3 uses `torchvision.models.mobilenet_v2` initialized with ImageNet weights. The final linear
classifier is replaced with a 30-output layer. The implementation retains the torchvision-style
`features` and `classifier` modules, which also keeps the model compatible with later Grad-CAM/XAI
work.

## Transfer-learning modes

### Frozen

- All feature blocks are frozen.
- Only the classifier is trainable.
- Frozen feature blocks, including their BatchNorm layers, remain in evaluation mode during
  training so running statistics do not change.
- AdamW classifier learning rate: `1e-3`.

### Partial

- Initialized from the validation-selected Frozen checkpoint at epoch 15.
- Feature blocks 0-14 remain frozen.
- Feature blocks 15-18 are trainable.
- The classifier remains trainable.
- BatchNorm in frozen blocks remains in evaluation mode; BatchNorm in blocks 15-18 trains normally.
- Differential AdamW learning rates:
  - blocks 15-18: `1e-5` for 1,526,080 parameters;
  - classifier: `1e-4` for 38,430 parameters.

## Parameter counts

| Mode | Total | Trainable | Frozen |
|---|---:|---:|---:|
| Frozen | 2,262,302 | 38,430 | 2,223,872 |
| Partial | 2,262,302 | 1,564,510 | 697,792 |

## Shared training settings

- Input: `224 x 224`
- Batch size: 32
- Epoch limit: 15
- Optimizer: AdamW
- Weight decay: `1e-4`
- Scheduler: CosineAnnealingLR
- AMP: enabled on CUDA
- Random seed: 42
- Early-stopping patience: 5
- Training split: 4,734 images
- Validation split: 1,022 images

Each epoch records train loss/Top-1, validation loss/Top-1/Top-5/Macro-F1, learning rate and epoch
time. The best checkpoint is selected solely by **validation Macro-F1**. Frozen selected epoch 15;
Partial selected epoch 14. The final test split was not loaded during either training run.
