# Data Layout

Keep raw datasets local under `data/raw/`. This folder is ignored by Git and should not be uploaded.

Tracked files can be placed here:

- `metadata/images.csv`: image paths and labels
- `metadata/class_to_idx.json`: class-name to integer-label mapping
- `splits/train.csv`: training split
- `splits/val.csv`: validation split
- `splits/test.csv`: test split
