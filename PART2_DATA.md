# PART2 Notes

## Data layout

This project uses the Leafsnap 30-subset dataset with lab and field images.

Data structure:

- `data/raw/`: raw dataset archive and extracted images (ignored by Git)
- `data/metadata/images.csv`: metadata rows with `path`, `label`, `class_idx`, and `source`
- `data/metadata/class_to_idx.json`: mapping from class name to numeric label
- `data/metadata/image_quality.csv`: image validity and size diagnostics
- `data/metadata/data_quality_summary.json`: summary statistics for Part 2 analysis
- `data/metadata/dataset_summary.csv`: species-level class counts by source
- `data/metadata/split_audit.csv`: per-split class counts for audit
- `data/splits/train.csv`: train split with canonical sample paths
- `data/splits/val.csv`: validation split with canonical sample paths
- `data/splits/test.csv`: test split with canonical sample paths

Part 2 specifics:

- Uses `field` and `lab/Auto_cropped` images only
- Builds a stratified group-aware split by species and stem ID
- Includes 30 species classes
- Total samples: 7,395 images
- Lab images: 5,320
- Field images: 2,075

## What I completed

- Parsed image metadata from the official list at `data/raw/5061353/leafsnap-dataset-30subset-images.txt`.
- Filtered the dataset to keep only `field` images and `lab/Auto_cropped` images.
- Generated `data/metadata/images.csv` with columns `path,label,class_idx,source`.
- Generated `data/metadata/class_to_idx.json` with mappings for 30 species.
- Created the dataset split files:
  - `data/splits/train.csv`
  - `data/splits/val.csv`
  - `data/splits/test.csv`
- Added audit summaries:
  - `data/metadata/dataset_summary.csv` with per-class field/lab/total counts
  - `data/metadata/split_audit.csv` with per-split class counts
- Performed data quality checks and output:
  - `data/metadata/image_quality.csv`
  - `data/metadata/data_quality_summary.json`
- Completed Part 2 EDA and saved visualizations to `report/figures/data/`.

## Current results

- Total samples: 7,395 (matches the official image list count exactly)
- Number of classes: 30
- Lab images: 5,320
- Field images: 2,075
- The train/val/test splits are non-overlapping (verified no shared leaf-sample group IDs across splits), and all split paths exist in `data/metadata/images.csv`.

### Fixed 2026-07-17: 156 missing lab images

An earlier version of `src/data/build_metadata.py` skipped any `lab`-source
row whose official path did not already point into `Auto_cropped/` (it hit a
bare `continue`). 156 rows for two species were affected:

- `Broussonettia papyrifera`: 7 images
- `Chionanthus virginicus`: 149 images

All 156 have a valid, readable `Auto_cropped` counterpart on disk at
`dataset/images/lab/Auto_cropped/<species>/<filename>`, and none duplicate an
already-included row. The parser now normalizes any non-`Auto_cropped` lab
path to its `Auto_cropped` equivalent instead of dropping it. Total sample
count went from 7,239 to 7,395, matching
`leafsnap-dataset-30subset-images.txt` exactly.

**This changes `data/metadata/images.csv` and all three split files.**
Anyone who trained against the previous split (Baseline, MobileNetV2,
ResNet18) needs to re-run using the regenerated splits before results are
comparable — the old checkpoints/metrics are not valid for the final
comparison table.

## What can be done next 

1. Use the existing split files for training:
   - `data/splits/train.csv`
   - `data/splits/val.csv`
   - `data/splits/test.csv`
2. Use the class mapping directly:
   - `data/metadata/class_to_idx.json`
3. For training, use the Part 1 script or other model code:
   - `python3 part1_custom_cnn.py --use-splits --split-dir data/splits --output-dir outputs/part2_aug`
4. Further analysis options:
   - training result comparison
   - augmentation comparison
   - error analysis
   - Grad-CAM / explainability analysis

## Notes

- I have finished Part 2 data cleaning and analysis.
- The dataset and splits were corrected on 2026-07-17 (see above) — anyone
  who already ran training against the old 7,239-image split must re-run
  against the current `data/splits` files before results are compared or
  reported.
