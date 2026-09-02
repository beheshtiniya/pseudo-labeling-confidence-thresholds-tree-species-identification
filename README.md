# Experiment 2: Confidence-Threshold Optimization and Evaluation Pipeline for Tree Species Detection

This repository contains the integrated implementation of **Experiment 2** for tree species detection using **Faster R-CNN with a ResNet-50-FPN backbone**.

The experiment investigates pseudo-label confidence-threshold selection using a **completely unlabeled image pool**, repeated independent training trials, and a common held-out validation set. The current integrated script combines pseudo-label generation, threshold-specific training, confusion-matrix evaluation, Precision/Recall/F1/Accuracy calculation, and mAP@0.5/Precision-Recall evaluation in a single workflow.

**Main script:** `experiment2_unlabeled_pool_40trials_final.py`

The previous standalone evaluation logic from `final_test.py` and `map.py` has been incorporated into the integrated script so that separate execution of those scripts is no longer required for the final Experiment 2 workflow.

---

## 1. Objective

The purpose of this experiment is to determine an appropriate confidence threshold for pseudo-label selection and to evaluate how threshold-dependent pseudo-labeling affects Faster R-CNN performance.

The experiment reports:

- macro-averaged **Precision**
- macro-averaged **Recall**
- macro-averaged **F1-score**
- **Accuracy**
- per-class **Average Precision (AP)**
- **mAP@0.5**
- per-class **Precision-Recall (PR) curves**
- **Confusion Matrices**
- mean and standard deviation across repeated trials

The preferred confidence threshold is selected using the **highest mean validation mAP@0.5 across repeated independent runs**.

The strongest individual run is also reported, but only as a descriptive result.

---

## 2. Experimental Design

Ten confidence thresholds are evaluated:

```text
0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.80, 0.85, 0.90, 0.95
```

Each threshold is evaluated using **4 independent repetitions**:

```text
10 thresholds × 4 repetitions = 40 independent trials
```

The primary threshold-selection criterion is:

\[
T_{\mathrm{conf}}^{\mathrm{best}}
=
\arg\max_{T_{\mathrm{conf}}}
\overline{\mathrm{mAP@0.5}}_{\mathrm{val}}
(T_{\mathrm{conf}})
\]

Precision, Recall, F1-score, and Accuracy are used as complementary validation metrics.

---

## 3. Overall Workflow

For each confidence threshold and each independent repetition, the following workflow is executed:

```text
Expert-labeled training set
        │
        ▼
Initialize Faster R-CNN
        │
        ▼
Train pseudo-label generator
        │
        ▼
Completely unlabeled pseudo-labeling pool
        │
        ▼
Generate candidate detections
        │
        ▼
Confidence filtering
conf(p) > T_conf
        │
        ▼
Class-wise NMS
IoU threshold = 0.50
        │
        ▼
Accepted pseudo-labels
        │
        ├──────────────────────┐
        │                      │
        ▼                      │
Original expert annotations   │
        │                      │
        └──────────┬───────────┘
                   ▼
Expanded training set
                   │
                   ▼
Reinitialize Faster R-CNN
                   │
                   ▼
Train threshold-specific model
                   │
                   ▼
Held-out validation set
                   │
                   ▼
Raw validation predictions
                   │
                   ▼
IoU-based prediction filtering
                   │
                   ▼
merged_predictions.csv
          ┌────────┴────────┐
          ▼                 ▼
Confusion Matrix       mAP@0.5 / PR curves
P / R / F1 / Acc.      per-class AP
          └────────┬────────┘
                   ▼
Repeat 4 times per threshold
                   │
                   ▼
Mean validation metrics
                   │
                   ▼
Select threshold with highest
mean validation mAP@0.5
```

---

## 4. Important Note About the Unlabeled Pseudo-Labeling Pool

The pseudo-labeling pool used in this experiment is **completely unlabeled**.

Therefore, ground-truth-aware overlap filtering cannot be applied during pseudo-label selection because expert bounding boxes do not exist for the pool images.

Candidate pseudo-labels are selected using:

1. confidence filtering,

\[
\operatorname{conf}(p)>T_{\mathrm{conf}},
\]

and

2. **class-wise non-maximum suppression (NMS)** with IoU threshold `0.50`.

NMS is used to suppress redundant overlapping predictions within the same predicted class and image.

Ground-truth-aware filtering is only meaningful for images that already contain expert or partial annotations and is therefore **not part of the pseudo-label selection procedure for this unlabeled pool**.

---

## 5. Required Directory Structure

Organize the project as follows:

```text
BASE_DATA_DIR/
│
├── images/
│   ├── train_image_001.tif
│   ├── train_image_002.tif
│   ├── validation_image_001.tif
│   └── ...
│
├── pseudo_pool/
│   ├── unlabeled_image_001.tif
│   ├── unlabeled_image_002.tif
│   └── ...
│
├── train_labels.csv
├── val_labels.csv
├── confusion_matrix.py
└── experiment2_unlabeled_pool_40trials_final.py
```

### `images/`

Contains the labeled images referenced by:

- `train_labels.csv`
- `val_labels.csv`

### `pseudo_pool/`

Contains the completely unlabeled images used for pseudo-label generation.

No annotation CSV is required for the pseudo-labeling pool.

The integrated script checks that pseudo-pool filenames do not overlap with the labeled training or validation images.

---

## 6. Annotation CSV Format

Both `train_labels.csv` and `val_labels.csv` must contain:

```csv
filename,class,xmin,ymin,xmax,ymax
image_001.tif,1,125,83,217,194
image_001.tif,3,241,96,331,210
image_002.tif,2,77,104,163,225
```

Required fields:

| Column | Description |
|---|---|
| `filename` | Image filename |
| `class` | Integer object-class ID |
| `xmin` | Left x-coordinate |
| `ymin` | Top y-coordinate |
| `xmax` | Right x-coordinate |
| `ymax` | Bottom y-coordinate |

The Faster R-CNN configuration uses:

```python
NUM_CLASSES = 5
```

This corresponds to the background class plus four tree-species object classes.

---

## 7. Dependencies

Install the required Python packages:

```bash
pip install pandas numpy torch torchvision pillow tqdm scikit-learn matplotlib
```

The repository must also contain:

```text
confusion_matrix.py
```

because the integrated script reuses:

```python
from confusion_matrix import compute_confusion_matrix
```

to preserve compatibility with the previously used confusion-matrix evaluation logic.

CUDA is selected automatically when available:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 8. Configuration

Update the main dataset directory before execution:

```python
BASE_DATA_DIR = r"E:\FASTRCNN\FASTRCNN\dataset\psudo_labeling"
```

The script expects:

```python
LABELED_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")
PSEUDO_POOL_DIR = os.path.join(BASE_DATA_DIR, "pseudo_pool")

TRAIN_LABELS = os.path.join(BASE_DATA_DIR, "train_labels.csv")
VAL_LABELS = os.path.join(BASE_DATA_DIR, "val_labels.csv")
```

### Confidence thresholds

```python
THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95
]
```

### Independent repetitions

```python
N_REPEATS = 4
```

### Training parameters

```python
GENERATOR_EPOCHS = 10
FINAL_EPOCHS = 10

BATCH_SIZE = 4

LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
```

These values should remain unchanged when reproducing the reported experiment unless the experimental protocol itself is intentionally changed.

### IoU parameters

```python
PSEUDO_NMS_IOU = 0.50
EVAL_IOU = 0.50
```

---

## 9. Faster R-CNN Architecture

The experiment uses:

**Faster R-CNN with a ResNet-50-FPN backbone**

initialized with torchvision pretrained weights:

```python
torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)
```

The detection head is replaced according to the number of classes:

```python
FastRCNNPredictor(in_features, NUM_CLASSES)
```

Each independent trial contains two training stages:

1. training of a pseudo-label generator on the expert-labeled training data;
2. reinitialization of Faster R-CNN followed by training on the threshold-specific expanded dataset.

The final model is therefore not simply a continuation of the pseudo-label generator.

---

## 10. Independent Trial Initialization

Each threshold is evaluated four times.

A deterministic seed is generated from the threshold and repetition number and is applied to Python, NumPy, PyTorch, and CUDA when available.

The repeated trials are used to estimate mean and standard-deviation performance and reduce dependence on a single model initialization.

---

# Pseudo-Labeling Pipeline

## 11. Training the Pseudo-Label Generator

At the beginning of each independent trial, Faster R-CNN is trained using only the original expert-labeled training data:

```text
images/
train_labels.csv
```

This trained model is used only as the pseudo-label generator for that trial.

---

## 12. Prediction on the Unlabeled Pool

The generator is applied to all images inside:

```text
pseudo_pool/
```

Raw detections are saved as:

```text
pool_predictions_raw.csv
```

Each prediction contains:

```text
filename
class
xmin
ymin
xmax
ymax
score
```

---

## 13. Confidence-Based Pseudo-Label Selection

For threshold \(T_{\mathrm{conf}}\), a prediction is retained only when:

\[
\operatorname{conf}(p)>T_{\mathrm{conf}}.
\]

The comparison is strict.

For example, when:

```text
T_conf = 0.70
```

a prediction with score `0.701` is retained, whereas a prediction with score exactly `0.700` is not.

---

## 14. Duplicate Suppression

Because the pseudo-labeling images do not contain expert annotations, candidate pseudo-labels cannot be compared with ground truth.

Instead, class-wise NMS is applied separately for each image and predicted class using:

```python
PSEUDO_NMS_IOU = 0.50
```

The surviving pseudo-labels are saved as:

```text
pseudo_labels_accepted.csv
```

---

## 15. Expanded Training Set

All original expert annotations are preserved.

The accepted pseudo-labels are appended to the original expert-labeled training set:

\[
D_{\mathrm{expanded}}
=
D_{\mathrm{expert}}
\cup
P_{\mathrm{accepted}}.
\]

The resulting annotation table is saved as:

```text
expanded_train_labels.csv
```

Original labeled images are loaded from:

```text
images/
```

and pseudo-labeled pool images are loaded from:

```text
pseudo_pool/
```

A new Faster R-CNN model is then initialized and trained on this expanded configuration.

The final checkpoint for each run is saved as:

```text
fasterrcnn_final.pth
```

---

# Integrated Evaluation Pipeline

## 16. Purpose of the Evaluation Pipeline

The evaluation stage analyzes the performance of every threshold-specific Faster R-CNN model on the **same held-out validation set**.

The integrated evaluation reproduces the main logic previously implemented by the standalone `final_test.py` and `map.py` scripts.

For every trained run, the integrated script performs:

1. validation prediction;
2. IoU-based prediction filtering;
3. merged prediction generation;
4. confusion-matrix calculation;
5. macro Precision, Recall, and F1-score calculation;
6. Accuracy calculation;
7. per-class AP calculation;
8. mAP@0.5 calculation;
9. per-class Precision-Recall curve generation.

Separate execution of `final_test.py` and `map.py` is therefore not required for the final integrated workflow.

---

## 17. Validation Prediction

The final model from each independent trial is evaluated using the common validation set defined by:

```text
val_labels.csv
```

and the corresponding validation images in:

```text
images/
```

The validation set is not used for pseudo-label generation or final model training.

Raw predictions are saved as:

```text
val_predictions_raw.csv
```

Each raw prediction contains the image filename, predicted class, bounding-box coordinates, and confidence score.

---

## 18. IoU-Based Filtering for Evaluation

The validation filtering procedure follows the earlier `final_test.py` logic.

For every ground-truth bounding box:

1. all predictions from the same image are examined;
2. predictions with IoU greater than `0.50` are considered;
3. the prediction with the highest IoU is selected;
4. if two candidates have the same IoU, the one with the higher confidence score is selected.

The filtering step is class-agnostic, consistent with the previous evaluation implementation. Therefore, a spatially matched prediction with an incorrect class can subsequently appear as an off-diagonal error in the confusion matrix.

The filtered predictions are combined into:

```text
merged_predictions.csv
```

This file is used by both the confusion-matrix evaluation and the mAP@0.5 evaluation to preserve consistency with the previous workflow.

---

## 19. Confusion Matrix

The integrated experiment reuses the existing:

```python
compute_confusion_matrix(...)
```

function from:

```text
confusion_matrix.py
```

using:

```text
val_labels.csv
merged_predictions.csv
```

as inputs.

For each run, the following files are saved:

```text
confusion_matrix.png
confusion_matrix_raw.csv
```

The current model definition contains four tree-species object classes plus the Faster R-CNN background class.

---

## 20. Precision, Recall, F1-Score, and Accuracy

The label pairs returned by `compute_confusion_matrix` are used to calculate the classification metrics.

### Macro Precision

```python
precision_score(
    true_labels,
    pred_labels,
    average="macro",
    labels=labels_present
)
```

### Macro Recall

```python
recall_score(
    true_labels,
    pred_labels,
    average="macro",
    labels=labels_present
)
```

### Macro F1-score

```python
f1_score(
    true_labels,
    pred_labels,
    average="macro",
    labels=labels_present
)
```

Macro averaging assigns equal weight to the evaluated classes instead of weighting them by class frequency.

### Accuracy

Accuracy is calculated using:

```python
accuracy_score(
    true_labels,
    pred_labels
)
```

Conceptually:

\[
\mathrm{Accuracy}
=
\frac{\text{number of correct predictions}}
{\text{total number of evaluated predictions}}.
\]

These four metrics are included in every run-level results file.

---

# Evaluation of mAP@0.5 and Precision-Recall Curves

## 21. Objective

The detection evaluation provides a fine-grained assessment of object-detection performance using:

- per-class **Precision-Recall curves**
- **Average Precision (AP)** for each tree-species class
- **mAP@0.5** as the overall detection metric

This evaluation is integrated directly into:

```text
experiment2_unlabeled_pool_40trials_final.py
```

and does not require separate execution of `map.py`.

---

## 22. Inputs for mAP Evaluation

The mAP calculation uses:

```text
val_labels.csv
merged_predictions.csv
```

where:

- `val_labels.csv` contains the validation ground-truth boxes;
- `merged_predictions.csv` contains the IoU-filtered predictions generated during the integrated validation pipeline.

This intentionally preserves the previously used `final_test.py` + `map.py` evaluation sequence for comparability.

---

## 23. Per-Class Detection Matching

For each tree-species class:

1. validation ground-truth boxes are grouped by image;
2. predictions of that class are sorted by confidence score in descending order;
3. every predicted box is compared with ground-truth boxes of the same class and image;
4. a valid detection requires:

\[
IoU \geq 0.50;
\]

5. each ground-truth bounding box can be matched only once;
6. duplicate or unmatched predictions are counted as false positives.

This produces the true-positive and false-positive sequences needed for Precision-Recall evaluation.

---

## 24. Precision-Recall Computation

For each class, cumulative true positives and false positives are calculated.

Precision is computed as:

\[
\mathrm{Precision}
=
\frac{TP}{TP+FP}
\]

and Recall as:

\[
\mathrm{Recall}
=
\frac{TP}{N_{\mathrm{GT}}},
\]

where \(N_{\mathrm{GT}}\) is the total number of ground-truth objects for that class.

The resulting Precision and Recall arrays define the class-specific PR curve.

---

## 25. Average Precision

Average Precision is calculated using the **11-point interpolation procedure** used in the previous `map.py` implementation.

The evaluated recall levels are:

\[
0.0,\ 0.1,\ 0.2,\ldots,\ 1.0.
\]

At each recall level, the maximum precision observed at or above that recall is used.

For class \(c\):

\[
AP_c
=
\frac{1}{11}
\sum_{r\in\{0,0.1,\ldots,1\}}
P_{\mathrm{interp}}(r).
\]

Per-class AP values are saved as:

```text
per_class_ap.csv
```

---

## 26. mAP@0.5

The final mAP@0.5 is calculated as the arithmetic mean of the AP values across the evaluated tree-species classes:

\[
\mathrm{mAP@0.5}
=
\frac{1}{C}
\sum_{c=1}^{C}AP_c.
\]

Precision-Recall curves are saved as:

```text
precision_recall_curves.png
```

The run-level mAP@0.5 is also stored in:

```text
metrics.csv
```

and subsequently included in the threshold-level summary.

---

# Results Aggregation and Threshold Selection

## 27. Run-Level Outputs

Every independent trial generates:

```text
pool_predictions_raw.csv
pseudo_labels_accepted.csv
expanded_train_labels.csv
fasterrcnn_final.pth
val_predictions_raw.csv
merged_predictions.csv
confusion_matrix.png
confusion_matrix_raw.csv
per_class_ap.csv
precision_recall_curves.png
metrics.csv
```

The `metrics.csv` file contains:

```text
Threshold
Repeat
Num_Pseudo_Labels
Precision
Recall
F1
Accuracy
mAP@0.5
```

---

## 28. Threshold-Level Summary

After all 40 trials are completed, the script calculates for each confidence threshold:

- number of completed runs;
- mean number of accepted pseudo-labels;
- mean Precision;
- mean Recall;
- mean F1-score;
- mean Accuracy;
- mean mAP@0.5;
- standard deviation of each metric;
- descriptive arithmetic mean of the five mean metrics.

The output is saved as:

```text
threshold_summary.csv
```

The script also verifies that exactly 40 trials were completed before producing the final summary.

---

## 29. Descriptive Average

For compatibility with the threshold-comparison table reported for Experiment 2, the script calculates:

\[
\mathrm{Descriptive\ Average}
=
\frac{
\mathrm{Precision}
+
\mathrm{Recall}
+
\mathrm{F1}
+
\mathrm{Accuracy}
+
\mathrm{mAP@0.5}
}{5}.
\]

Each component in this expression is the threshold-level mean across four repetitions.

This value is included only as a **descriptive aggregate**.

It is not the primary threshold-selection criterion.

---

## 30. Preferred Confidence Threshold

The preferred threshold is selected according to the highest **mean validation mAP@0.5**:

\[
T_{\mathrm{conf}}^{\mathrm{best}}
=
\arg\max_{T_{\mathrm{conf}}}
\overline{\mathrm{mAP@0.5}}_{\mathrm{val}}
(T_{\mathrm{conf}}).
\]

This approach gives priority to repeated-run performance rather than to a single favorable model initialization.

---

## 31. Best Individual Run

The 40 independent trials are also ranked descriptively.

The best individual run is ranked primarily by:

```text
mAP@0.5
```

with F1-score used as a tie-breaker.

It is saved as:

```text
best_individual_run_descriptive.csv
```

This result is **not used to select the preferred threshold**.

The best individual result for each threshold is also saved as:

```text
best_individual_per_threshold_descriptive.csv
```

---

## 32. Four Repetitions per Threshold

The script automatically creates one file containing the four official runs for each threshold:

```text
threshold_0.50_four_runs.csv
threshold_0.55_four_runs.csv
threshold_0.60_four_runs.csv
threshold_0.65_four_runs.csv
threshold_0.70_four_runs.csv
threshold_0.75_four_runs.csv
threshold_0.80_four_runs.csv
threshold_0.85_four_runs.csv
threshold_0.90_four_runs.csv
threshold_0.95_four_runs.csv
```

The four runs corresponding to the automatically selected threshold are additionally saved as:

```text
selected_threshold_four_runs.csv
```

These files can be used directly when preparing threshold-specific manuscript tables.

---

## 33. Main Output Directory

All experiment outputs are written under:

```text
experiment2_unlabeled_pool/
```

Example:

```text
experiment2_unlabeled_pool/
│
├── th_0.5_run_1/
│   ├── pool_predictions_raw.csv
│   ├── pseudo_labels_accepted.csv
│   ├── expanded_train_labels.csv
│   ├── fasterrcnn_final.pth
│   ├── val_predictions_raw.csv
│   ├── merged_predictions.csv
│   ├── confusion_matrix.png
│   ├── confusion_matrix_raw.csv
│   ├── per_class_ap.csv
│   ├── precision_recall_curves.png
│   └── metrics.csv
│
├── th_0.5_run_2/
├── ...
├── th_0.95_run_4/
│
├── all_runs_metrics.csv
├── threshold_summary.csv
├── threshold_0.50_four_runs.csv
├── ...
├── threshold_0.95_four_runs.csv
├── selected_threshold_four_runs.csv
├── best_individual_run_descriptive.csv
└── best_individual_per_threshold_descriptive.csv
```

---

## 34. Files Used for Manuscript Results

| Manuscript result | Integrated-script output |
|---|---|
| Mean metrics for all thresholds | `threshold_summary.csv` |
| All 40 independent trials | `all_runs_metrics.csv` |
| Four runs at a specific threshold | `threshold_XX_four_runs.csv` |
| Four runs at selected threshold | `selected_threshold_four_runs.csv` |
| Best individual trial | `best_individual_run_descriptive.csv` |
| Best individual result per threshold | `best_individual_per_threshold_descriptive.csv` |
| Confusion Matrix | `confusion_matrix_raw.csv`, `confusion_matrix.png` |
| Per-class AP | `per_class_ap.csv` |
| PR curves | `precision_recall_curves.png` |
| Run-level complete metrics | `metrics.csv` |

---

## 35. Data-Separation Checks

Before running the experiment, the script checks that the completely unlabeled pseudo-labeling pool does not overlap with:

- the labeled training set;
- the held-out validation set.

If an overlapping filename is detected, execution stops.

The intended data separation is therefore:

```text
Expert-labeled training data
           ≠
Unlabeled pseudo-labeling pool
           ≠
Held-out validation data
```

---

## 36. Running the Experiment

Run:

```bash
python experiment2_unlabeled_pool_40trials_final.py
```

The program automatically evaluates all 10 confidence thresholds and all four repetitions.

No separate `final_test.py` or `map.py` execution is required for the integrated final workflow.

---

## 37. Relationship to the Previous Standalone Evaluation Scripts

The integrated implementation preserves the core behavior of the earlier evaluation scripts while consolidating the experiment into a single executable workflow.

### Previous `final_test.py` logic retained

- validation prediction with Faster R-CNN;
- IoU-based filtering at `0.50`;
- selection of the highest-IoU prediction for each ground-truth object;
- confidence as the tie-breaker;
- generation of `merged_predictions.csv`;
- use of `compute_confusion_matrix`;
- macro Precision;
- macro Recall;
- macro F1-score.

The integrated implementation additionally calculates **Accuracy**.

### Previous `map.py` logic retained

- class-wise evaluation;
- confidence-ranked detections;
- IoU threshold `0.50`;
- one-to-one prediction/ground-truth matching;
- cumulative TP and FP;
- Precision-Recall curves;
- 11-point interpolated AP;
- mean AP across classes.

The legacy standalone scripts may be retained in a `legacy/` directory for historical reproducibility, but they are not required to run the final integrated Experiment 2 implementation.

---

## 38. Interpretation of Mean and Best-Observed Results

Two types of results are intentionally distinguished.

### Mean threshold performance

This is the primary basis for threshold comparison.

Each threshold is represented by the mean of four independent repetitions.

### Best-observed individual performance

This describes the strongest individual model observed during the 40-trial threshold search.

It is reported only as supplementary descriptive evidence.

A strong isolated trial does not override the threshold selected from mean repeated-run performance.

---

## 39. Reproducibility Checklist

To reproduce the final Experiment 2 workflow:

1. use the same `train_labels.csv`;
2. use the same `val_labels.csv`;
3. use the same unlabeled `pseudo_pool/`;
4. evaluate all 10 thresholds from `0.50` to `0.95`;
5. run four independent repetitions per threshold;
6. preserve the same training hyperparameters;
7. keep `PSEUDO_NMS_IOU = 0.50`;
8. keep validation matching at `IoU = 0.50`;
9. use the same `confusion_matrix.py`;
10. retain macro Precision, Recall, and F1;
11. calculate Accuracy from the same confusion-matrix label pairs;
12. retain the 11-point AP implementation;
13. calculate mAP@0.5 from `merged_predictions.csv` to preserve the previous evaluation sequence;
14. select the preferred threshold from mean validation mAP@0.5;
15. treat best individual runs as descriptive only.

---

## 40. Summary

The final integrated script provides a complete confidence-threshold optimization and evaluation pipeline for Faster R-CNN-based tree species detection.

The main characteristics are:

- a completely unlabeled pseudo-labeling pool;
- 10 confidence thresholds from `0.50` to `0.95`;
- four independent repetitions per threshold;
- 40 total trials;
- confidence-based pseudo-label selection;
- class-wise NMS at IoU `0.50`;
- preservation of all original expert annotations;
- independent Faster R-CNN reinitialization for threshold-specific training;
- evaluation on a common held-out validation set;
- confusion matrices;
- macro Precision, Recall, and F1-score;
- Accuracy;
- per-class AP;
- 11-point mAP@0.5;
- per-class PR curves;
- threshold selection using mean validation mAP@0.5;
- best individual runs retained only for descriptive reporting.

This integrated workflow replaces the need to execute the former standalone evaluation scripts separately and provides a single reproducible pipeline for the final Experiment 2 analysis.

---

## License

MIT License
