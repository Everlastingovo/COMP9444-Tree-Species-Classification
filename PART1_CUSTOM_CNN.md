# Part 1: Custom CNN Baseline

This part answers the E0 research question:

> When no pretrained model is used, what baseline performance can a simple deep learning model achieve?

The implementation is organised in the project framework:

- `configs/baseline.yaml`: baseline experiment settings
- `src/models/baseline_cnn.py`: custom CNN architecture
- `src/data/`: dataset loading, metadata, and split helpers
- `src/training/trainer.py`: training loop
- `part1_custom_cnn.py`: simple command-line entry point

The baseline model does not load pretrained weights.

## Setup

Use Python 3.12, then install the required packages:

```powershell
py -3.12 -m pip install -r requirements.txt
```

The dataset zip should stay local and should not be uploaded to GitHub:

```text
data/raw/5061353/leafsnap-dataset-30subset.zip
```

If you already extracted the dataset somewhere else, pass the extracted `dataset/images` folder directly:

```powershell
py -3.12 part1_custom_cnn.py --images-root "D:\path\to\leafsnap-dataset-30subset\dataset\images" --epochs 1 --max-batches 2
```

## Run a Quick Test

This only checks that the code and dataset loading work:

```powershell
py -3.12 part1_custom_cnn.py --epochs 1 --max-batches 2
```

## Run the Main Experiment

```powershell
py -3.12 part1_custom_cnn.py --epochs 20 --batch-size 32 --image-source field
```

Outputs are saved under:

```text
outputs/baseline/
```

Important output files:

- `best_custom_cnn.pt`: best validation checkpoint
- `metrics.csv`: training and validation loss/accuracy per epoch
- `summary.json`: final result summary, including test accuracy
- `dataset_split.csv`: reproducible train/validation/test split

The reusable metadata and split files are also written to:

```text
data/metadata/
data/splits/
```

Raw images remain under `data/raw/` and are ignored by Git.

## Current Baseline Result

The first completed CPU run used 30 field-image classes:

```text
Train images: 1447
Validation images: 314
Test images: 314
Best validation accuracy: 66.24%
Test accuracy: 68.47%
Test loss: 1.0337
```

## Report Text Draft

For the baseline experiment, we trained a custom convolutional neural network from scratch without using pretrained weights. The model contains four convolutional blocks with batch normalization, ReLU activation, max pooling, adaptive average pooling, and a small fully connected classifier. The dataset was split into training, validation, and test subsets using a stratified split so that each tree species was represented in each split. Basic image augmentation, including random crop, horizontal flip, brightness adjustment, and contrast adjustment, was applied only to the training set.

This experiment provides the baseline performance for tree species classification when no external pretrained visual features are used. The validation accuracy was used to select the best checkpoint, and the final test accuracy was reported from that checkpoint.
