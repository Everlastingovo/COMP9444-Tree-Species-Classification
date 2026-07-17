# Part 2: Data Preparation and EDA

## Locked data foundation

The project uses the Leafsnap 30-species subset with both field images and cleaned
`lab/Auto_cropped` images. Raw lab originals are never included alongside their cleaned versions.

Current final counts:

- Official manifest paths: 7,395
- Unique SHA-256 content hashes before conflict exclusion: 6,823
- Same-class duplicate paths removed: 400
- Cross-class conflict groups excluded: 66 hashes / 238 images
- Final unique-content images: 6,757
- Classes: 30
- Lab images: 5,108
- Field images: 1,649
- Train / validation / test: 4,734 / 1,022 / 1,001

All models must reuse the checked-in split CSVs. Results trained on the earlier 7,239-image or
7,395-path splits are not comparable with this final content-cleaned split.

## Data layout

- `data/metadata/images.csv`: final `path,label,class_idx,source,sha256` rows
- `data/metadata/class_to_idx.json`: fixed mapping for classes 0-29
- `data/metadata/duplicate_same_class.csv`: removed same-label duplicates
- `data/metadata/conflicting_labels.csv`: excluded cross-label hash groups
- `data/metadata/image_quality.csv`: hash, dimensions, mode and aspect ratio
- `data/metadata/data_quality_summary.json`: count reconciliation
- `data/metadata/dataset_summary.csv`: final per-class/source counts
- `data/metadata/split_audit.csv`: final per-split/class counts
- `data/splits/{train,val,test}.csv`: locked split files

All paths in metadata and split files are relative to the Leafsnap subset root. The standard root is
`data/raw/5061353/leafsnap-dataset-30subset`; another location can be supplied with `--data-root`
or `LEAFSNAP_DATA_ROOT` as documented in `data/README.md`.

## Cleaning rules

`src/data/build_metadata.py` first normalizes any official lab path missing `Auto_cropped` to the
corresponding cleaned path. It then computes SHA-256 for all 7,395 normalized official images.

For one-class hash groups, the first official-manifest path is canonical and later paths are removed.
For multi-class hash groups, no label is inferred: every member is excluded and written to the
conflict audit. This is why the final count is 6,757 rather than 6,823.

The count identity is:

```text
7,395 official paths - 400 same-class duplicate paths - 238 conflicting images = 6,757
```

## Split policy

`part2_prepare_metadata.py` uses `stratified_group_split` with validation ratio 0.15, test ratio
0.15 and seed 42. The group key retains known multiple-shot leaf samples in one split. After
content cleaning:

- every class appears in train, validation and test;
- path overlap across splits is zero;
- SHA-256 content overlap across splits is zero;
- no final SHA-256 maps to more than one class.

## Rebuild command

Do not rebuild the locked files casually. If the team intentionally regenerates them, use the same
official dataset and command:

```powershell
python part2_prepare_metadata.py --data-root "D:\path\to\leafsnap-dataset-30subset"
```

Then run the integrity tests described in `data/README.md` and review both duplicate/conflict audit
CSVs before accepting the new split.

## EDA

`notebooks/01_data_eda.ipynb` and `part2_data_eda.py` cover class distribution, source distribution,
image dimensions, lab/field examples and augmentation examples. `part2_data_eda.py` accepts the same
`--data-root` option so metadata paths remain machine-independent.

## Change history

### Restored 156 cleaned lab images

The earlier parser silently dropped 156 official lab rows whose paths omitted `Auto_cropped`. Those
rows are now normalized to their readable cleaned counterparts; raw ruler/color-card images are not
used.

### Removed exact-content leakage and label conflicts

The 7,395 official paths contained 466 repeated-content hash groups. Same-class duplicates are now
collapsed to the first manifest path, while 66 multi-class groups are excluded for manual audit.
The resulting 6,757-image split has no exact-content leakage.
