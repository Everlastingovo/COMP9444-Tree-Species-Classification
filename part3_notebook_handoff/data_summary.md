# Data summary

## Locked dataset

| Split | Images | Lab | Field | Classes |
|---|---:|---:|---:|---:|
| Train | 4,734 | 3,568 | 1,166 | 30 |
| Validation | 1,022 | 780 | 242 | 30 |
| Test | 1,001 | 760 | 241 | 30 |
| Total | 6,757 | 5,108 | 1,649 | 30 |

The authoritative class mapping is `data/metadata/class_to_idx.json`, with contiguous indices
0-29. `data/statistics/split_audit.csv` provides per-class split counts, and
`data/statistics/dataset_summary.csv` provides per-class lab/field counts.

## Dataset interface

`LeafDataset` returns `(image, target)`:

- Batched image shape: `[batch, 3, 224, 224]`
- Image dtype: `torch.float32`
- Batched label shape: `[batch]`
- Label dtype: `torch.int64`
- Label range: `0-29`

All split paths are portable relative paths. Lab samples use only
`dataset/images/lab/Auto_cropped/...`; raw lab photographs containing the scale and colour card are
not part of the locked dataset.

## Preprocessing

MobileNetV2 uses ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training preprocessing:

1. Contain the image within `248 x 248` (`224 + 24`).
2. Pad to a square with a white background.
3. Random crop to `224 x 224`.
4. Horizontal mirror with probability 0.5.
5. Brightness adjustment with probability 0.35 and factor 0.80-1.20.
6. Contrast adjustment with probability 0.35 and factor 0.80-1.20.
7. Convert to tensor and apply ImageNet normalization.

Validation and test preprocessing is deterministic: contain within `224 x 224`, white square
padding, resize to `224 x 224`, tensor conversion and ImageNet normalization. It uses no random
crop, flip, colour augmentation, TTA or ensemble.

## Integrity conclusions

The locked metadata contains 6,757 readable images and 30 classes. The Part 2 integrity suite
confirmed:

- no missing or invalid metadata image paths;
- no path overlap among train, validation and test;
- no SHA-256 content overlap among splits;
- one unique content hash per final sample;
- no remaining cross-class hash conflict;
- class indices are exactly 0-29;
- every class appears in every split;
- a `224 x 224` DataLoader batch has finite tensors and valid `torch.int64` labels.

The split uses seed 42, class stratification and filename-prefix grouping to reduce known
multiple-shot leaf leakage. Byte-level hashing cannot prove that visually similar files with
different bytes depict different physical leaves; this remains a documented limitation.

No raw images are included in this handoff.
