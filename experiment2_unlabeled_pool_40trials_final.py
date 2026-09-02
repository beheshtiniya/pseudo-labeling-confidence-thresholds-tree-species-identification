# -*- coding: utf-8 -*-
"""
Experiment 2 - Confidence-threshold search using a COMPLETELY UNLABELED
pseudo-labeling pool.

Workflow
--------
1) Train an independently initialized Faster R-CNN on expert-labeled train data.
2) Apply the trained generator to the unlabeled pseudo-labeling pool.
3) Keep predictions with conf(p) > T_conf.
4) Remove duplicate predictions inside the unlabeled pool using class-wise NMS.
   NOTE: Ground-truth-aware filtering CANNOT be performed on a truly unlabeled pool.
5) Preserve all original expert annotations and add accepted pseudo-labels.
6) Reinitialize Faster R-CNN and train independently on the expanded dataset.
7) Evaluate on validation data using the same evaluation logic supplied by the user:
   - IoU-based filtering for confusion-matrix evaluation
   - confusion_matrix.compute_confusion_matrix(...)
   - macro Precision, Recall, F1
   - Accuracy
   - 11-point AP / mAP@0.5 logic from map.py
8) Evaluate 10 confidence thresholds from 0.50 to 0.95 in steps of 0.05.
9) Repeat 4 independent trials per confidence threshold (40 trials total).
10) Compare thresholds using mean validation mAP@0.5; other metrics are complementary.
11) Report the best individual run descriptively only.

Expected structure
------------------
BASE_DATA_DIR/
    images/                 # labeled train/validation images
    pseudo_pool/            # completely unlabeled pool images
    train_labels.csv
    val_labels.csv
    confusion_matrix.py     # user's existing implementation

Label CSV columns:
    filename,class,xmin,ymin,xmax,ymax

IMPORTANT
---------
If the manuscript says that pseudo-labels from the UNLABELED pool underwent
"ground-truth-aware overlap filtering", that wording must be changed because
there is no ground truth for these images. GT-aware overlap filtering is only
possible for partially/pre-labeled images.
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.ops import box_iou, nms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt

# Reuse the user's own confusion-matrix implementation exactly.
from confusion_matrix import compute_confusion_matrix


# ============================================================
# CONFIG
# ============================================================

BASE_DATA_DIR = r"E:\FASTRCNN\FASTRCNN\dataset\psudo_labeling"
LABELED_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")
PSEUDO_POOL_DIR = os.path.join(BASE_DATA_DIR, "pseudo_pool")

TRAIN_LABELS = os.path.join(BASE_DATA_DIR, "train_labels.csv")
VAL_LABELS = os.path.join(BASE_DATA_DIR, "val_labels.csv")

OUTPUT_ROOT = os.path.join(BASE_DATA_DIR, "experiment2_unlabeled_pool")

# Replace with exactly the threshold set used in your experiment.
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_REPEATS = 4
EXPECTED_TOTAL_TRIALS = len(THRESHOLDS) * N_REPEATS  # 10 x 4 = 40

NUM_CLASSES = 5  # background + four tree classes

# Training hyperparameters: set to the exact values used in the paper.
GENERATOR_EPOCHS = 10
FINAL_EPOCHS = 10
BATCH_SIZE = 4
NUM_WORKERS = 0

LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Duplicate removal among pseudo-labels.
# This is NOT ground-truth-aware filtering.
PSEUDO_NMS_IOU = 0.50

# Validation matching threshold, consistent with supplied codes.
EVAL_IOU = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

VALID_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

print("Device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def read_label_csv(path):
    df = pd.read_csv(path)
    required = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[required].copy()
    df["filename"] = df["filename"].astype(str)
    df["class"] = df["class"].astype(int)
    return df


def list_pool_images():
    if not os.path.isdir(PSEUDO_POOL_DIR):
        raise FileNotFoundError(
            f"Unlabeled pseudo-labeling pool folder not found:\n{PSEUDO_POOL_DIR}"
        )

    files = [
        f for f in os.listdir(PSEUDO_POOL_DIR)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ]

    if not files:
        raise RuntimeError("No unlabeled images found in pseudo_pool/.")

    return sorted(files)


class DetectionDataset(Dataset):
    def __init__(self, images_dir, labels_df):
        self.images_dir = images_dir
        self.labels_df = labels_df.copy()
        self.labels_df["filename"] = self.labels_df["filename"].astype(str)
        self.image_files = self.labels_df["filename"].drop_duplicates().tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image = Image.open(
            os.path.join(self.images_dir, filename)
        ).convert("RGB")
        image = F.to_tensor(image)

        rows = self.labels_df[self.labels_df["filename"] == filename]

        boxes = torch.tensor(
            rows[["xmin", "ymin", "xmax", "ymax"]].values,
            dtype=torch.float32
        )
        labels = torch.tensor(
            rows["class"].values,
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target, filename


class UnlabeledPoolDataset(Dataset):
    def __init__(self, pool_dir, filenames):
        self.pool_dir = pool_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(
            os.path.join(self.pool_dir, filename)
        ).convert("RGB")
        image = F.to_tensor(image)
        return image, filename


def detection_collate(batch):
    return tuple(zip(*batch))


def pool_collate(batch):
    images, filenames = zip(*batch)
    return list(images), list(filenames)


# ============================================================
# MODEL / TRAINING
# ============================================================

def build_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )
    return model.to(DEVICE)


def train_model(model, loader, epochs):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for images, targets, _ in tqdm(
            loader,
            desc=f"Training {epoch}/{epochs}",
            leave=False
        ):
            images = [img.to(DEVICE) for img in images]
            targets = [
                {k: v.to(DEVICE) for k, v in target.items()}
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(
            f"epoch={epoch:02d}, "
            f"loss={np.mean(losses) if losses else 0:.6f}"
        )


# ============================================================
# UNLABELED POOL PSEUDO-LABELING
# ============================================================

def generate_pool_predictions(model, pool_loader):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, filenames in tqdm(
            pool_loader,
            desc="Pseudo-labeling unlabeled pool",
            leave=False
        ):
            outputs = model([img.to(DEVICE) for img in images])

            for filename, output in zip(filenames, outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    rows.append({
                        "filename": str(filename),
                        "class": int(label),
                        "xmin": float(box[0]),
                        "ymin": float(box[1]),
                        "xmax": float(box[2]),
                        "ymax": float(box[3]),
                        "score": float(score),
                    })

    return pd.DataFrame(
        rows,
        columns=[
            "filename", "class",
            "xmin", "ymin", "xmax", "ymax",
            "score"
        ]
    )


def select_pseudo_labels_unlabeled_pool(
    prediction_df,
    confidence_threshold,
    nms_iou=0.50
):
    """
    Selection for a COMPLETELY UNLABELED pool.

    1) conf(p) > T_conf
    2) class-wise NMS within each image to suppress duplicate predictions.

    No GT-aware overlap test is performed because no expert boxes exist.
    """
    df = prediction_df[
        prediction_df["score"] > confidence_threshold
    ].copy()

    accepted_groups = []

    for filename, image_df in df.groupby("filename"):
        for cls, class_df in image_df.groupby("class"):
            boxes = torch.tensor(
                class_df[["xmin", "ymin", "xmax", "ymax"]].values,
                dtype=torch.float32
            )
            scores = torch.tensor(
                class_df["score"].values,
                dtype=torch.float32
            )

            keep_indices = nms(boxes, scores, nms_iou).cpu().numpy()
            accepted_groups.append(
                class_df.iloc[keep_indices].copy()
            )

    if not accepted_groups:
        return pd.DataFrame(columns=df.columns)

    accepted = pd.concat(
        accepted_groups,
        ignore_index=True
    )

    return accepted


def build_expanded_training_table(train_df, accepted_pseudo_df):
    """
    The pool images have no expert annotations.

    Original expert annotations are preserved.
    Accepted pseudo-labels are appended as annotations for pool images.
    """
    pseudo_labels = accepted_pseudo_df[
        ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
    ].copy()

    return pd.concat(
        [train_df, pseudo_labels],
        ignore_index=True
    )


class MixedDetectionDataset(Dataset):
    """
    Reads labeled training images from LABELED_IMAGES_DIR
    and pseudo-labeled pool images from PSEUDO_POOL_DIR.
    """

    def __init__(self, labels_df, original_train_filenames):
        self.labels_df = labels_df.copy()
        self.labels_df["filename"] = self.labels_df["filename"].astype(str)
        self.files = self.labels_df["filename"].drop_duplicates().tolist()
        self.original_train_filenames = set(
            map(str, original_train_filenames)
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        if filename in self.original_train_filenames:
            path = os.path.join(LABELED_IMAGES_DIR, filename)
        else:
            path = os.path.join(PSEUDO_POOL_DIR, filename)

        image = Image.open(path).convert("RGB")
        image = F.to_tensor(image)

        rows = self.labels_df[
            self.labels_df["filename"] == filename
        ]

        boxes = torch.tensor(
            rows[["xmin", "ymin", "xmax", "ymax"]].values,
            dtype=torch.float32
        )
        labels = torch.tensor(
            rows["class"].values,
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target, filename


# ============================================================
# VALIDATION PREDICTION
# ============================================================

def predict_validation(model, val_loader):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, _, filenames in tqdm(
            val_loader,
            desc="Validation prediction",
            leave=False
        ):
            outputs = model([img.to(DEVICE) for img in images])

            for filename, output in zip(filenames, outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    rows.append({
                        "filename": str(filename),
                        "class": int(label),
                        "xmin": float(box[0]),
                        "ymin": float(box[1]),
                        "xmax": float(box[2]),
                        "ymax": float(box[3]),
                        "score": float(score),
                    })

    return pd.DataFrame(
        rows,
        columns=[
            "filename", "class",
            "xmin", "ymin", "xmax", "ymax",
            "score"
        ]
    )


# ============================================================
# SAME IoU FILTERING LOGIC AS USER'S final_test.py
# ============================================================

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = (
        max(0, inter_x_max - inter_x_min)
        * max(0, inter_y_max - inter_y_min)
    )

    area1 = (
        (x1_max - x1_min)
        * (y1_max - y1_min)
    )
    area2 = (
        (x2_max - x2_min)
        * (y2_max - y2_min)
    )

    return inter_area / (
        area1 + area2 - inter_area + 1e-6
    )


def filter_predictions_like_final_test(gt_df, pred_df):
    """
    Mirrors the user's final_test.py:
    For each GT object, choose the prediction with IoU > 0.5 having:
      1) highest IoU,
      2) highest score as tie breaker.

    Matching is intentionally class-agnostic, as in the supplied code.
    """
    output_rows = []

    for filename, gt_boxes in gt_df.groupby("filename"):
        image_pred = pred_df[
            pred_df["filename"].astype(str) == str(filename)
        ]

        for _, gt_row in gt_boxes.iterrows():
            gt_box = [
                gt_row["xmin"], gt_row["ymin"],
                gt_row["xmax"], gt_row["ymax"]
            ]

            best_box = None
            best_iou = -1
            best_score = -1

            for _, box_row in image_pred.iterrows():
                pred_box = [
                    box_row["xmin"], box_row["ymin"],
                    box_row["xmax"], box_row["ymax"]
                ]

                score = box_row["score"]
                iou = calculate_iou(gt_box, pred_box)

                if (
                    iou > EVAL_IOU
                    and (
                        iou > best_iou
                        or (
                            iou == best_iou
                            and score > best_score
                        )
                    )
                ):
                    best_box = box_row
                    best_iou = iou
                    best_score = score

            if best_box is not None:
                output_rows.append(best_box.to_dict())

    return pd.DataFrame(output_rows)


# ============================================================
# CONFUSION MATRIX + P/R/F1/ACCURACY
# ============================================================

def evaluate_confusion_metrics(
    gt_csv_path,
    merged_prediction_path,
    output_dir
):
    """
    Uses user's existing compute_confusion_matrix() exactly,
    then adds Accuracy alongside the same macro Precision/Recall/F1 style.
    """
    conf_matrix, true_labels, pred_labels = compute_confusion_matrix(
        gt_csv_path,
        merged_prediction_path
    )

    labels_present = sorted(np.unique(true_labels))

    precision = precision_score(
        true_labels,
        pred_labels,
        average="macro",
        labels=labels_present,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        pred_labels,
        average="macro",
        labels=labels_present,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        pred_labels,
        average="macro",
        labels=labels_present,
        zero_division=0
    )

    accuracy = accuracy_score(
        true_labels,
        pred_labels
    )

    # Raw confusion matrix
    labels = list(range(NUM_CLASSES))
    pd.DataFrame(
        conf_matrix,
        index=labels,
        columns=labels
    ).to_csv(
        os.path.join(
            output_dir,
            "confusion_matrix_raw.csv"
        ),
        index_label="True/Pred"
    )

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(conf_matrix)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j, i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png"
        ),
        dpi=200
    )
    plt.close(fig)

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
    }


# ============================================================
# mAP@0.5 - SAME LOGIC AS USER'S map.py
# ============================================================

def compute_map50_like_user_code(
    gt_df,
    pred_df,
    iou_threshold=0.5
):
    all_classes = set(
        gt_df["class"]
    ).union(
        set(pred_df["class"])
    )

    aps = []
    per_class_ap = {}
    pr_curves = {}

    gt_grouped = gt_df.groupby("filename")

    for cls in sorted(all_classes):
        true_positives = []
        total_gt_boxes = 0
        gt_boxes_per_image = defaultdict(list)

        for filename, group in gt_grouped:
            boxes = group[
                group["class"] == cls
            ][
                ["xmin", "ymin", "xmax", "ymax"]
            ].values

            if len(boxes) > 0:
                gt_boxes_per_image[filename] = {
                    "boxes": torch.tensor(
                        boxes,
                        dtype=torch.float32
                    ),
                    "detected": [False] * len(boxes)
                }
                total_gt_boxes += len(boxes)

        pred_data = pred_df[
            pred_df["class"] == cls
        ].sort_values(
            by="score",
            ascending=False
        )

        for _, row in pred_data.iterrows():
            filename = row["filename"]

            pred_box = torch.tensor(
                [[
                    row["xmin"], row["ymin"],
                    row["xmax"], row["ymax"]
                ]],
                dtype=torch.float32
            )

            if filename in gt_boxes_per_image:
                gt_info = gt_boxes_per_image[filename]

                ious = box_iou(
                    pred_box,
                    gt_info["boxes"]
                )[0]

                max_iou, max_idx = torch.max(
                    ious,
                    dim=0
                )

                max_idx = int(max_idx.item())

                if (
                    max_iou >= iou_threshold
                    and not gt_info["detected"][max_idx]
                ):
                    true_positives.append(1)
                    gt_info["detected"][max_idx] = True
                else:
                    true_positives.append(0)
            else:
                true_positives.append(0)

        tp_cumsum = np.cumsum(
            true_positives
        )
        fp_cumsum = np.cumsum(
            [1 - x for x in true_positives]
        )

        precisions = (
            tp_cumsum
            / (
                tp_cumsum
                + fp_cumsum
                + 1e-6
            )
        )

        recalls = (
            tp_cumsum
            / (
                total_gt_boxes
                + 1e-6
            )
        )

        pr_curves[cls] = (
            recalls,
            precisions
        )

        ap = 0.0

        for t in np.linspace(0, 1, 11):
            precisions_at_recall = (
                precisions[
                    recalls >= t
                ]
            )

            p = (
                max(precisions_at_recall)
                if len(precisions_at_recall) > 0
                else 0
            )

            ap += p / 11.0

        aps.append(ap)
        per_class_ap[int(cls)] = float(ap)

    map_50 = (
        float(np.mean(aps))
        if aps
        else 0.0
    )

    return (
        map_50,
        per_class_ap,
        pr_curves
    )


def save_pr_curves(pr_curves, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for cls, (
        recalls,
        precisions
    ) in pr_curves.items():
        ax.plot(
            recalls,
            precisions,
            label=f"Class {cls}"
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Precision-Recall Curves (per class)"
    )
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# ONE REPEAT
# ============================================================

def run_one_repeat(
    threshold,
    repeat_idx,
    train_df,
    val_df,
    pool_files
):
    seed = (
        50000
        + int(threshold * 1000)
        + repeat_idx
    )
    set_seed(seed)

    run_name = (
        f"th_{threshold}_run_{repeat_idx}"
    )

    run_dir = os.path.join(
        OUTPUT_ROOT,
        run_name
    )
    os.makedirs(
        run_dir,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(run_name)
    print("=" * 70)

    # --------------------------------------------------------
    # 1) Independent pseudo-label generator
    # --------------------------------------------------------
    train_dataset = DetectionDataset(
        LABELED_IMAGES_DIR,
        train_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate
    )

    generator = build_model()

    print("Training pseudo-label generator...")
    train_model(
        generator,
        train_loader,
        GENERATOR_EPOCHS
    )

    # --------------------------------------------------------
    # 2) Unlabeled pool predictions
    # --------------------------------------------------------
    pool_dataset = UnlabeledPoolDataset(
        PSEUDO_POOL_DIR,
        pool_files
    )

    pool_loader = DataLoader(
        pool_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=pool_collate
    )

    raw_pool_predictions = (
        generate_pool_predictions(
            generator,
            pool_loader
        )
    )

    raw_pool_predictions.to_csv(
        os.path.join(
            run_dir,
            "pool_predictions_raw.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 3) Confidence threshold + NMS
    # --------------------------------------------------------
    accepted_pseudo = (
        select_pseudo_labels_unlabeled_pool(
            raw_pool_predictions,
            confidence_threshold=threshold,
            nms_iou=PSEUDO_NMS_IOU
        )
    )

    accepted_pseudo.to_csv(
        os.path.join(
            run_dir,
            "pseudo_labels_accepted.csv"
        ),
        index=False
    )

    print(
        "Accepted pseudo-labels:",
        len(accepted_pseudo)
    )

    # --------------------------------------------------------
    # 4) Expanded Dataset 1 configuration
    # --------------------------------------------------------
    expanded_df = build_expanded_training_table(
        train_df,
        accepted_pseudo
    )

    expanded_df.to_csv(
        os.path.join(
            run_dir,
            "expanded_train_labels.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 5) Reinitialize and retrain independently
    # --------------------------------------------------------
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_model = build_model()

    expanded_dataset = MixedDetectionDataset(
        expanded_df,
        original_train_filenames=train_df[
            "filename"
        ].unique()
    )

    expanded_loader = DataLoader(
        expanded_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate
    )

    print("Training expanded configuration...")
    train_model(
        final_model,
        expanded_loader,
        FINAL_EPOCHS
    )

    torch.save(
        final_model.state_dict(),
        os.path.join(
            run_dir,
            "fasterrcnn_final.pth"
        )
    )

    # --------------------------------------------------------
    # 6) Raw validation predictions
    # --------------------------------------------------------
    val_dataset = DetectionDataset(
        LABELED_IMAGES_DIR,
        val_df
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate
    )

    raw_val_predictions = predict_validation(
        final_model,
        val_loader
    )

    raw_val_path = os.path.join(
        run_dir,
        "val_predictions_raw.csv"
    )

    raw_val_predictions.to_csv(
        raw_val_path,
        index=False
    )

    # --------------------------------------------------------
    # 7) final_test.py-style IoU filtering
    # --------------------------------------------------------
    merged_predictions = (
        filter_predictions_like_final_test(
            val_df,
            raw_val_predictions
        )
    )

    merged_path = os.path.join(
        run_dir,
        "merged_predictions.csv"
    )

    merged_predictions.to_csv(
        merged_path,
        index=False
    )

    # --------------------------------------------------------
    # 8) Confusion + P/R/F1 + Accuracy
    # --------------------------------------------------------
    cls_metrics = evaluate_confusion_metrics(
        VAL_LABELS,
        merged_path,
        run_dir
    )

    # --------------------------------------------------------
    # 9) map.py-style mAP@0.5
    #    To reproduce the previous workflow exactly, mAP is
    #    computed from merged_predictions.csv.
    # --------------------------------------------------------
    map50, per_class_ap, pr_curves = (
        compute_map50_like_user_code(
            val_df,
            merged_predictions,
            iou_threshold=EVAL_IOU
        )
    )

    pd.DataFrame(
        [
            {
                "Class": cls,
                "AP@0.5": ap
            }
            for cls, ap
            in per_class_ap.items()
        ]
    ).to_csv(
        os.path.join(
            run_dir,
            "per_class_ap.csv"
        ),
        index=False
    )

    save_pr_curves(
        pr_curves,
        os.path.join(
            run_dir,
            "precision_recall_curves.png"
        )
    )

    result = {
        "Threshold": threshold,
        "Repeat": repeat_idx,
        "Num_Pseudo_Labels": len(
            accepted_pseudo
        ),
        "Precision": cls_metrics[
            "Precision"
        ],
        "Recall": cls_metrics[
            "Recall"
        ],
        "F1": cls_metrics["F1"],
        "Accuracy": cls_metrics[
            "Accuracy"
        ],
        "mAP@0.5": map50,
    }

    pd.DataFrame(
        [result]
    ).to_csv(
        os.path.join(
            run_dir,
            "metrics.csv"
        ),
        index=False
    )

    print(
        pd.DataFrame(
            [result]
        ).to_string(
            index=False
        )
    )

    del final_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    train_df = read_label_csv(
        TRAIN_LABELS
    )
    val_df = read_label_csv(
        VAL_LABELS
    )
    pool_files = list_pool_images()

    # Strong separation checks.
    train_files = set(
        train_df["filename"].astype(str)
    )
    val_files = set(
        val_df["filename"].astype(str)
    )
    pool_set = set(pool_files)

    if train_files & pool_set:
        raise ValueError(
            "Pseudo-labeling pool overlaps labeled training images."
        )

    if val_files & pool_set:
        raise ValueError(
            "Pseudo-labeling pool overlaps validation images."
        )

    all_results = []

    for threshold in THRESHOLDS:
        for repeat_idx in range(
            1,
            N_REPEATS + 1
        ):
            result = run_one_repeat(
                threshold,
                repeat_idx,
                train_df,
                val_df,
                pool_files
            )
            all_results.append(result)

    all_df = pd.DataFrame(
        all_results
    )

    # The Results section reports 10 thresholds x 4 repetitions = 40 trials.
    if len(all_df) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_TRIALS} completed trials, "
            f"but found {len(all_df)}."
        )

    all_df.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "all_runs_metrics.csv"
        ),
        index=False
    )

    # Save the four official repetitions for each threshold separately.
    # These files can be used directly to construct tables such as the
    # four T=0.70 runs in the Results section.
    for threshold in THRESHOLDS:
        threshold_runs = (
            all_df[
                np.isclose(
                    all_df["Threshold"].astype(float),
                    float(threshold)
                )
            ]
            .sort_values("Repeat")
            .reset_index(drop=True)
        )
        threshold_runs.to_csv(
            os.path.join(
                OUTPUT_ROOT,
                f"threshold_{threshold:.2f}_four_runs.csv"
            ),
            index=False
        )

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "mAP@0.5"
    ]

    mean_df = (
        all_df.groupby(
            "Threshold"
        )[metrics]
        .mean()
        .add_suffix("_Mean")
    )

    std_df = (
        all_df.groupby(
            "Threshold"
        )[metrics]
        .std(ddof=1)
        .add_suffix("_Std")
    )

    pseudo_mean = (
        all_df.groupby(
            "Threshold"
        )[
            "Num_Pseudo_Labels"
        ]
        .mean()
        .rename(
            "Mean_Num_Pseudo_Labels"
        )
    )

    n_runs = (
        all_df.groupby(
            "Threshold"
        )["Repeat"]
        .count()
        .rename("N_Runs")
    )

    summary = pd.concat(
        [
            n_runs,
            pseudo_mean,
            mean_df,
            std_df
        ],
        axis=1
    ).reset_index()

    # Descriptive aggregate used in Table 6:
    # unweighted arithmetic mean of the five MEAN validation metrics.
    summary["Descriptive_Average"] = summary[
        [
            "Precision_Mean",
            "Recall_Mean",
            "F1_Mean",
            "Accuracy_Mean",
            "mAP@0.5_Mean"
        ]
    ].mean(axis=1)

    # Primary threshold selection:
    # highest MEAN validation mAP@0.5 across the four independent repetitions.
    best_idx = summary[
        "mAP@0.5_Mean"
    ].idxmax()

    best_threshold = float(
        summary.loc[
            best_idx,
            "Threshold"
        ]
    )

    summary[
        "Selected_Best_Threshold"
    ] = np.isclose(
        summary["Threshold"].astype(float),
        best_threshold
    )

    # Put columns in a results-friendly order.
    summary = summary[
        [
            "Threshold",
            "N_Runs",
            "Mean_Num_Pseudo_Labels",
            "Precision_Mean",
            "Recall_Mean",
            "F1_Mean",
            "Accuracy_Mean",
            "mAP@0.5_Mean",
            "Descriptive_Average",
            "Precision_Std",
            "Recall_Std",
            "F1_Std",
            "Accuracy_Std",
            "mAP@0.5_Std",
            "Selected_Best_Threshold"
        ]
    ]

    summary.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "threshold_summary.csv"
        ),
        index=False
    )

    # Best individual run is DESCRIPTIVE ONLY and is not used to select T_best.
    # Ranking is primarily by mAP@0.5, with F1 as the tie-breaker.
    ranked_runs = (
        all_df
        .sort_values(
            by=["mAP@0.5", "F1"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )

    best_individual = ranked_runs.iloc[[0]].copy()
    best_individual["Role"] = "Descriptive only"

    best_individual.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "best_individual_run_descriptive.csv"
        ),
        index=False
    )

    # Descriptive top individual run from each threshold.
    best_per_threshold = (
        all_df
        .sort_values(
            by=["Threshold", "mAP@0.5", "F1"],
            ascending=[True, False, False]
        )
        .groupby("Threshold", as_index=False)
        .first()
        .sort_values(
            by=["mAP@0.5", "F1"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    best_per_threshold["Descriptive_Rank"] = (
        np.arange(1, len(best_per_threshold) + 1)
    )

    best_per_threshold.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "best_individual_per_threshold_descriptive.csv"
        ),
        index=False
    )

    # The four official runs at the selected threshold.
    selected_threshold_runs = (
        all_df[
            np.isclose(
                all_df["Threshold"].astype(float),
                best_threshold
            )
        ]
        .sort_values("Repeat")
        .reset_index(drop=True)
    )

    selected_threshold_runs.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            "selected_threshold_four_runs.csv"
        ),
        index=False
    )

    print("\n" + "=" * 80)
    print(f"COMPLETED TRIALS: {len(all_df)} / {EXPECTED_TOTAL_TRIALS}")
    print("=" * 80)

    print("\nThreshold summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSelected threshold from highest mean validation mAP@0.5:",
        best_threshold
    )

    print("\nBest individual run (descriptive only):")
    print(
        best_individual.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
