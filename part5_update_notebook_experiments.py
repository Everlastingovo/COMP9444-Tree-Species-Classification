import argparse
from pathlib import Path

import nbformat


SECTION_TAG = "part5-comparison-experiments"
VISUAL_TAG = "part5-report-visuals"

MARKDOWN_BY_HEADING = {
    "# 06 Evaluation and Explainability": """# 06 Evaluation and Explainability

This notebook presents the final, post-selection evaluation of the locked
Baseline Custom CNN, partially fine-tuned MobileNetV2, and partially fine-tuned
ResNet18. All models use the corrected 30-class Leafsnap subset and the same
1,001-image test split. The notebook does not train, tune, or select a model
from test performance.""",
    "## 1. Setup": """## 1. Setup

The analysis reads immutable checkpoints, saved per-image predictions, and the
shared class mapping. Generated tables and figures are stored under
`outputs/comparison/`; report-ready copies are stored under `reports/`. The
checkpoint hashes recorded in the handoff manifests are used to identify the
evaluated models.""",
    "## 2. Locked Evaluation Inputs": """## 2. Locked Evaluation Inputs

The corrected dataset contains 6,757 images divided into 4,734 training, 1,022
validation, and 1,001 test images. The test split contains 760 laboratory and
241 field images. All three handoffs use the same contiguous 30-class mapping
and the same ordered test paths. Model selection was completed using validation
performance before any final test analysis.""",
    "## 3. Unified Metrics": """## 3. Unified Metrics

Accuracy, Macro-F1, Weighted-F1, Top-5 accuracy, per-class metrics, and
lab/field metrics are recomputed from traceable per-image predictions.

**Reporting policy.** Headline test accuracy uses the model owner's locked
result: 80.52% for the Baseline, 93.81% for MobileNetV2, and 97.10% for
ResNet18. Detailed Baseline analysis uses the locally regenerated prediction
file, which gives 80.42% and differs by one image. The brightness experiment is
a separate local post-hoc rerun and measures every change relative to that
model's local brightness-1.0 result. These values are not interchangeable.""",
    "## 4. Model Comparison": """## 4. Model Comparison

ResNet18 is the strongest overall model, reaching 97.10% Accuracy, 97.22%
Macro-F1, 97.11% Weighted-F1, and 99.80% Top-5 accuracy. MobileNetV2 ranks
second with 93.81% Accuracy and 93.99% Macro-F1, while the Baseline reaches an
owner-reported 80.52% Accuracy. The close agreement between Accuracy and
Macro-F1 for both transfer models indicates that their gains are not confined
to only the largest classes. ResNet18 is therefore selected as the final model;
MobileNetV2 remains a substantially smaller alternative.""",
    "## 5. Confusion Matrix and Per-class Results": """## 5. Confusion Matrix and Per-class Results

The final ResNet18 model correctly classifies 972 of 1,001 test images. Its
weakest class is `ulmus_americana` (F1 = 85.71%), followed by
`diospyros_virginiana` (92.50%) and `ostrya_virginiana` (92.54%). The strongest
two-way confusion is `ulmus_americana` versus `ulmus_rubra`, with five errors
in total. Four additional `ulmus_americana` images are predicted as
`ostrya_virginiana`. These errors are consistent with fine-grained visual
similarity among simple, serrated leaves rather than broad failure across the
30 classes.""",
    "## 6. Lab vs Field and Illumination Robustness": """## 6. Lab vs Field and Illumination Robustness

The Baseline falls from 82.63% lab accuracy to 73.44% field accuracy, a
9.19-point gap, showing limited robustness to real backgrounds and acquisition
conditions. ResNet18 retains 97.50% lab and 95.85% field accuracy, reducing the
gap to 1.65 points. MobileNetV2 records 93.03% lab and 96.27% field accuracy.
The apparent field advantage for MobileNetV2 should not be treated as proof
that field images are easier because the field subset contains only 241 images
and 29 represented classes, compared with 760 lab images and all 30 classes.""",
    "## 7. Model Efficiency": """## 7. Model Efficiency

The shared benchmark uses one Apple CPU, eight Torch threads, ten warm-up
iterations, and 50 measured batch-one iterations. It times only the forward
pass and retains each model's deployed input size. The Baseline is smallest and
fastest (1.21M parameters, 4.87 MB, 7.73 ms) but has much lower accuracy.
MobileNetV2 uses 2.26M parameters and a 9.30 MB checkpoint, while ResNet18 uses
11.19M parameters and 44.85 MB. On this CPU, MobileNetV2 takes 57.85 ms per
image compared with 17.52 ms for ResNet18. This hardware-specific result shows
that fewer parameters and depthwise convolutions do not guarantee lower
latency on every backend.""",
    "## 8. Brightness Robustness": """## 8. Brightness Robustness

Global RGB brightness factors of 0.6, 0.8, 1.0, 1.2, and 1.4 are applied before
the locked resize and normalization. At factor 0.6, Baseline accuracy collapses
from 80.42% to 50.05%. Its maximum loss is 30.37 percentage points. In contrast,
MobileNetV2 retains 92.01% at factor 0.6 and has a maximum drop of 2.80 points;
ResNet18 retains 94.61% and has a maximum drop of 2.50 points. The transfer
models are therefore substantially more stable under this controlled global
brightness shift. Synthetic brightness does not model spatial shadows, glare,
sensor noise, or color-temperature changes, so it complements rather than
replaces the lab/field analysis.""",
    "## 9. Error Analysis": """## 9. Error Analysis

The highest-confidence errors reveal two recurring patterns. First, visually
similar taxa remain difficult: all three models confuse members of the
`Ulmus` group with each other or with `ostrya_virginiana`. ResNet18 predicts one
`ulmus_americana` image as `ostrya_virginiana` with 99.55% confidence and one
`ulmus_pumila` field image as the same class with 94.28% confidence. Second,
scale and framing matter. A laboratory `liriodendron_tulipifera` image contains
only a small partial leaf near the image boundary and is confidently
misclassified by all three models. These high-confidence errors indicate that
confidence alone is not a reliable failure detector and motivate calibration
or uncertainty-aware review.""",
    "## 10. Grad-CAM": """## 10. Grad-CAM

Grad-CAM explains the predicted class using `model.features[3][3]` for the
Baseline, `model.features[-1]` for MobileNetV2, and `model.layer4[-1]` for
ResNet18. Correct examples generally activate over the leaf body, outline,
central vein, and compound-leaf arrangement rather than only the surrounding
background. In incorrect `Ulmus` examples, the heatmaps still cover meaningful
leaf regions, suggesting that the model sees the object but maps shared
morphology to the wrong species. The cropped `liriodendron_tulipifera` example
shows attention split between the small visible leaf and image boundaries,
illustrating sensitivity to framing. Grad-CAM is qualitative evidence of
attention location, not proof that a highlighted feature causally determines
the prediction.""",
    "## 11. Discussion": """## 11. Discussion

Transfer learning produces the main performance gain. ResNet18 improves
headline accuracy by 16.58 points over the Baseline and by 3.30 points over
MobileNetV2 while maintaining the strongest Macro-F1 and field result. Its
residual representation appears better able to preserve fine-grained shape and
texture cues under source and brightness changes. MobileNetV2 offers a much
smaller checkpoint with only a modest accuracy reduction, making it attractive
when storage matters, although it is not the fastest model on the tested CPU.

The remaining errors are concentrated in morphologically similar species,
especially `Ulmus`, rather than spread uniformly across classes. The Baseline's
large lab-to-field gap and severe darkening failure suggest greater dependence
on the controlled acquisition style. The transfer models generalize more
consistently, but their high-confidence mistakes show that deployment should
still expose confidence and support expert review for ambiguous leaves.""",
    "## 12. Limitations and Future Work": """## 12. Limitations and Future Work

The test set is internal to the same Leafsnap subset used for model
development, so external geographic and device generalization remains unknown.
Only 241 field images and 29 field classes are represented, which limits source
comparisons. Global brightness is a simplified illumination proxy and does not
cover shadows, glare, white-balance shifts, blur, occlusion, or seasonal
variation. CPU latency is measured on one machine and cannot predict mobile,
CUDA, or optimized-runtime performance. Grad-CAM examples are qualitative and
do not establish causal feature use. Finally, cross-platform inference changes
one Baseline and one ResNet18 clean prediction, so owner-reported headline
metrics and local stress-test baselines are kept explicitly separate.

Future work should collect a larger external field set, evaluate calibration,
add controlled shadow and color-shift tests, investigate class-balanced or
metric-learning objectives for similar species, and benchmark exported models
on the intended deployment hardware.""",
    "## 13. Conclusion": """## 13. Conclusion

ResNet18 is the recommended final model. It achieves 97.10% test accuracy,
97.22% Macro-F1, 95.85% field accuracy, and the smallest observed maximum
brightness drop at 2.50 percentage points. MobileNetV2 provides a compact
alternative at 93.81% accuracy, while the Baseline is smaller and faster but
substantially less accurate and less robust. The principal unresolved failure
mode is confident confusion among visually similar species, especially
`Ulmus`, which should guide future data collection and uncertainty-aware
deployment.""",
}

UNIFIED_METRICS_CODE = """import csv

summary_paths = sorted(COMPARISON_DIR.glob("*/analysis_summary.json"))
model_summaries = {}
for summary_path in summary_paths:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_summaries[summary["model"]] = summary

with (COMPARISON_DIR / "model_comparison.csv").open(
    "r", encoding="utf-8", newline=""
) as csv_file:
    comparison_rows = list(csv.DictReader(csv_file))

print(
    f"{'model':<26} {'headline_acc':>14} {'local_acc':>12} "
    f"{'macro_f1':>12} {'weighted_f1':>14} {'top5':>10}"
)
for row in comparison_rows:
    print(
        f"{row['model']:<26} "
        f"{float(row['owner_reported_accuracy']):>14.4f} "
        f"{float(row['accuracy']):>12.4f} "
        f"{float(row['macro_f1']):>12.4f} "
        f"{float(row['weighted_f1']):>14.4f} "
        f"{float(row['top5_accuracy']):>10.4f}"
    )"""


def tagged_cell(cell_type: str, source: str):
    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source)
    else:
        cell = nbformat.v4.new_code_cell(source)
    cell.metadata["part5_section"] = SECTION_TAG
    return cell


def visual_cell(source: str):
    cell = nbformat.v4.new_code_cell(source)
    cell.metadata["part5_section"] = VISUAL_TAG
    return cell


def update_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if cell.metadata.get("part5_section") not in {SECTION_TAG, VISUAL_TAG}
    ]

    heading_updates = {
        "## 7. Error Analysis": "## 9. Error Analysis",
        "## 8. Grad-CAM": "## 10. Grad-CAM",
        "## 9. Discussion": "## 11. Discussion",
        "## 10. Limitations and Future Work": (
            "## 12. Limitations and Future Work"
        ),
        "## 11. Conclusion": "## 13. Conclusion",
    }
    for cell in notebook.cells:
        lines = cell.source.splitlines()
        if lines and lines[0] in heading_updates:
            lines[0] = heading_updates[lines[0]]
            cell.source = "\n".join(lines)
        lines = cell.source.splitlines()
        if lines and lines[0] in MARKDOWN_BY_HEADING:
            cell.source = MARKDOWN_BY_HEADING[lines[0]]
        if cell.cell_type == "code" and cell.source.lstrip().startswith(
            "summary_paths ="
        ):
            cell.source = UNIFIED_METRICS_CODE

    insertion_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.source.splitlines()
        and cell.source.splitlines()[0] == "## 9. Error Analysis"
    )
    experiment_cells = [
        tagged_cell(
            "markdown",
            """## 7. Model Efficiency

Compare parameter count, checkpoint size, batch-one latency, and batch
throughput under one shared CPU protocol. Timings cover the model forward pass
only; they exclude image decoding, preprocessing, and file I/O. Each model
retains its locked deployment input resolution.""",
        ),
        tagged_cell(
            "code",
            """import csv
from IPython.display import Image, display

EXPERIMENT_DIR = COMPARISON_DIR / "experiments"

with (EXPERIMENT_DIR / "model_efficiency.csv").open(
    "r", encoding="utf-8", newline=""
) as csv_file:
    efficiency_rows = list(csv.DictReader(csv_file))

print(
    f"{'model':<26} {'params':>10} {'size_mb':>10} "
    f"{'batch1_ms':>12} {'batch32_img_s':>16}"
)
for row in efficiency_rows:
    print(
        f"{row['model']:<26} "
        f"{int(row['total_parameters']) / 1_000_000:>9.2f}M "
        f"{float(row['checkpoint_size_mb']):>10.2f} "
        f"{float(row['batch1_median_latency_ms']):>12.2f} "
        f"{float(row['batch_throughput_images_per_second']):>16.2f}"
    )

display(Image(filename=str(EXPERIMENT_DIR / "model_efficiency.png")))""",
        ),
        tagged_cell(
            "markdown",
            """## 8. Brightness Robustness

Apply deterministic global RGB brightness factors of 0.6, 0.8, 1.0, 1.2,
and 1.4 to every locked test image before the model-specific resize and
normalization. Checkpoints remain frozen. Report changes relative to each
model's local 1.0 run and retain lab/field results. This is a controlled stress
test, not a complete simulation of real-world illumination.""",
        ),
        tagged_cell(
            "code",
            """with (EXPERIMENT_DIR / "brightness_robustness_summary.csv").open(
    "r", encoding="utf-8", newline=""
) as csv_file:
    brightness_summary = list(csv.DictReader(csv_file))

print(
    f"{'model':<26} {'clean':>9} {'perturbed_mean':>16} "
    f"{'worst_factor':>14} {'max_drop':>10}"
)
for row in brightness_summary:
    print(
        f"{row['model']:<26} "
        f"{float(row['clean_accuracy']):>9.4f} "
        f"{float(row['mean_perturbed_accuracy']):>16.4f} "
        f"{float(row['worst_brightness_factor']):>14.1f} "
        f"{float(row['maximum_accuracy_drop']):>10.4f}"
    )

display(Image(filename=str(EXPERIMENT_DIR / "brightness_robustness.png")))
display(
    Image(filename=str(EXPERIMENT_DIR / "brightness_source_robustness.png"))
)""",
        ),
    ]
    notebook.cells[insertion_index:insertion_index] = experiment_cells

    insert_visual_after_heading(
        notebook,
        "## 4. Model Comparison",
        visual_cell(
            """REPORT_FIGURE_DIR = REPO_ROOT / "reports" / "figures"
display(NotebookImage(filename=str(REPORT_FIGURE_DIR / "model_comparison.png")))
display(NotebookImage(filename=str(REPORT_FIGURE_DIR / "source_comparison.png")))"""
        ),
    )
    insert_visual_after_heading(
        notebook,
        "## 9. Error Analysis",
        visual_cell(
            """display(
    NotebookImage(filename=str(REPORT_FIGURE_DIR / "resnet18_error_cases.png"))
)"""
        ),
    )
    insert_visual_after_heading(
        notebook,
        "## 10. Grad-CAM",
        visual_cell(
            """for model_name in ("baseline", "mobilenetv2", "resnet18"):
    print(f"{model_name} Grad-CAM")
    display(
        NotebookImage(
            filename=str(
                REPORT_FIGURE_DIR / f"{model_name}_gradcam_examples.png"
            )
        )
    )"""
        ),
    )
    nbformat.write(notebook, path)


def insert_visual_after_heading(notebook, heading: str, cell) -> None:
    heading_index = next(
        index
        for index, candidate in enumerate(notebook.cells)
        if candidate.source.splitlines()
        and candidate.source.splitlines()[0] == heading
    )
    notebook.cells.insert(heading_index + 1, cell)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert the efficiency and brightness sections in Part 5."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/06_evaluation_xai.ipynb"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_notebook(args.notebook)
    print(f"Updated {args.notebook}")


if __name__ == "__main__":
    main()
