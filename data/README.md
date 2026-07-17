# Data

## Source

Leafsnap 30-species subset, downloaded from https://zenodo.org/records/5061353 and extracted
under `data/raw/5061353/leafsnap-dataset-30subset/` (git-ignored — not committed). The official
image manifest is `data/raw/5061353/leafsnap-dataset-30subset-images.txt`; every valid sample in
this project traces back to a row in that file. Two capture sources are present:

- `lab`: pressed leaves photographed under controlled studio lighting against a plain background
  with a color/scale reference card.
- `field`: leaves photographed outdoors on a phone camera, with natural backgrounds, variable
  lighting, and occasional occlusion/blur.

## Valid paths

`lab` samples are only ever taken from the `Auto_cropped` (cleaned) subfolder, never the raw
pressed-leaf originals, to avoid double-counting the same physical leaf as two samples. The
official manifest does not always spell out the `Auto_cropped` segment in its path column —
`src/data/build_metadata.py` normalizes any `lab` path missing that segment to its `Auto_cropped`
equivalent (`dataset/images/lab/Auto_cropped/<species>/<file>.jpg`) rather than dropping it (see
`PART2_DATA.md` for the bug this fixed: 156 images were previously being silently discarded this
way). `field` paths are used as listed.

## Classes and sample counts

- 30 species classes, mapping fixed in `data/metadata/class_to_idx.json`.
- 7,395 total valid images — matches the official manifest count exactly.
  - `lab`: 5,320 (72.0%)
  - `field`: 2,075 (28.0%)
- Per-class counts range from 198 (`gleditsia_triacanthos`) to 448 (`maclura_pomifera`), a ~2.3x
  imbalance — see `data/metadata/dataset_summary.csv` for the full per-class field/lab breakdown.

## Split

70% train / 15% val / 15% test, stratified by class and grouped so that multiple photos of the
same physical leaf (same filename prefix before the trailing `-N` shot index) never end up split
across train/val/test — see `src/data/split_dataset.py::stratified_group_split`. Seed is fixed at
42.

- train: 5,179
- val: 1,120
- test: 1,096

These are locked in `data/splits/{train,val,test}.csv`. **Every model in this project must load
these exact files (`--use-splits --split-dir data/splits`) rather than re-deriving its own split**
— otherwise results across models are not comparable. Per-split, per-class counts are in
`data/metadata/split_audit.csv`; verified to have zero path-level and zero leaf-group-level
overlap across the three splits.

## Files in this directory

| File | Description |
|---|---|
| `metadata/images.csv` | Every valid sample: `path,label,class_idx,source` |
| `metadata/class_to_idx.json` | Fixed 30-class mapping, read-only for all other members |
| `metadata/dataset_summary.csv` | Per-class total/field/lab counts |
| `metadata/split_audit.csv` | Per-split, per-class counts |
| `metadata/image_quality.csv` | Per-image width/height/mode/aspect ratio |
| `metadata/data_quality_summary.json` | Dataset-level quality summary |
| `metadata/augmentation_ablation.csv` | No-augmentation vs augmentation ablation results (see `notebooks/02_augmentation_ablation.ipynb`) |
| `splits/train.csv`, `splits/val.csv`, `splits/test.csv` | Locked train/val/test sample lists |
| `raw/` | Downloaded dataset archive and extracted images (git-ignored) |

## Known limitations

- Class imbalance (2.3x between smallest and largest class) is moderate but present — macro-F1,
  not just accuracy, should be reported, and per-class recall for the smaller classes is worth
  checking during error analysis.
- `lab` images dominate the dataset (72%) while the project's stated goal is field robustness.
  Overall accuracy alone can look good while field accuracy lags — always report lab/field
  accuracy separately, not just an overall number.
- The group-aware split prevents the same leaf sample (same filename prefix) from crossing splits,
  but it cannot detect duplicate photos of the same physical leaf taken under a different filename
  convention; none were found by manual spot-checking, but this wasn't exhaustively verified beyond
  the filename-prefix heuristic.
