# Data

## Source

This project uses the Leafsnap 30-species subset from the Zenodo record referenced by the project.
Raw data stays local and is ignored by Git. Every candidate sample starts from a row in the official
`leafsnap-dataset-30subset-images.txt` manifest.

Two capture sources are present:

- `lab`: pressed leaves photographed under controlled conditions.
- `field`: phone images with natural backgrounds, variable lighting and possible occlusion/blur.

## Valid lab paths

Lab samples are only taken from `dataset/images/lab/Auto_cropped/`, never from the corresponding
raw pressed-leaf originals. Some official manifest rows omit the `Auto_cropped` segment;
`src/data/build_metadata.py` normalizes those rows to their cleaned counterpart before hashing.
This preserves the 156 images previously dropped by the old parser without double-counting raw and
cleaned versions.

## Content-level cleaning

Every normalized official image is hashed with SHA-256 before metadata or splits are written.

- Same hash and same class: retain the first path in official-manifest order and record each removed
  path in `metadata/duplicate_same_class.csv`.
- Same hash and different classes: do not guess or change a label; exclude the whole hash group and
  record every member in `metadata/conflicting_labels.csv`.

The count reconciliation is:

- Official manifest paths: 7,395
- Unique content hashes before conflict exclusion: 6,823
- Same-class duplicate paths removed: 400
- Conflicting hash groups excluded: 66 groups / 238 images
- Final unique-content samples: 6,757

## Classes and sample counts

- 30 species, with the fixed mapping in `metadata/class_to_idx.json`.
- 6,757 final images:
  - `lab`: 5,108
  - `field`: 1,649
- Per-class counts are in `metadata/dataset_summary.csv`.

## Locked split

The split is approximately 70% train / 15% validation / 15% test, stratified by class and grouped
by the filename prefix before the trailing numeric shot index. Seed is fixed at 42.

- train: 4,734
- validation: 1,022
- test: 1,001

Every class appears in every split. The three split files have zero path overlap and zero SHA-256
content overlap. Models must reuse `splits/train.csv`, `splits/val.csv` and `splits/test.csv` rather
than deriving new splits.

## Data root and portable paths

Metadata and split paths are relative to the Leafsnap subset root and begin with `dataset/`. The
default subset root is `data/raw/5061353/leafsnap-dataset-30subset`.

For another local location, pass `--data-root` to Part 2 scripts:

```powershell
python part2_prepare_metadata.py --data-root "D:\path\to\leafsnap-dataset-30subset"
python part2_data_eda.py --data-root "D:\path\to\leafsnap-dataset-30subset"
```

CSV loading also accepts a `data_root` argument. Existing callers that do not pass it can configure
the root through `LEAFSNAP_DATA_ROOT`:

```powershell
$env:LEAFSNAP_DATA_ROOT = "D:\path\to\leafsnap-dataset-30subset"
```

## Files

| File | Description |
|---|---|
| `metadata/images.csv` | Final samples: `path,label,class_idx,source,sha256` |
| `metadata/class_to_idx.json` | Fixed 30-class mapping |
| `metadata/duplicate_same_class.csv` | Removed same-class duplicate paths and canonical path |
| `metadata/conflicting_labels.csv` | Excluded cross-class content conflicts |
| `metadata/dataset_summary.csv` | Final per-class field/lab counts |
| `metadata/split_audit.csv` | Final per-split, per-class counts |
| `metadata/image_quality.csv` | Per-image hash, dimensions, mode and aspect ratio |
| `metadata/data_quality_summary.json` | Count reconciliation and data-quality summary |
| `splits/train.csv`, `splits/val.csv`, `splits/test.csv` | Locked split sample lists |
| `raw/` | Local archive and extracted images; ignored by Git |

## Validation

With the dataset available, run the lightweight integrity tests with the root configured:

```powershell
$env:LEAFSNAP_DATA_ROOT = "D:\path\to\leafsnap-dataset-30subset"
python -m pytest -q -p no:cacheprovider tests/test_part2_data_integrity.py
```

## Known limitations

- Class and source imbalance remain; report macro-F1 and separate lab/field performance.
- SHA-256 removes byte-identical duplicates, and filename grouping protects known multiple-shot
  samples. These checks cannot prove that different bytes and unrelated filenames represent
  different physical leaves; visually near-duplicate detection remains outside Part 2 scope.
