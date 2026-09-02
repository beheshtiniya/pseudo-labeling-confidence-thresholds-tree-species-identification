import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def compute_confusion_matrix(true_csv_path, pred_csv_path):
    df_pred = pd.read_csv(pred_csv_path)
    df_true = pd.read_csv(true_csv_path)

    # مرتب‌سازی بر اساس filename
    df_pred = df_pred.sort_values(by=['filename']).reset_index(drop=True)
    df_true = df_true.sort_values(by=['filename']).reset_index(drop=True)

    new_true_labels = []
    new_pred_labels = []

    for filename in df_true['filename'].unique():
        true_boxes = df_true[df_true['filename'] == filename]
        pred_boxes = df_pred[df_pred['filename'] == filename]
        used_pred = set()

        for _, true_row in true_boxes.iterrows():
            best_iou = 0
            best_pred_idx = None

            for pred_idx, pred_row in pred_boxes.iterrows():
                if pred_idx in used_pred:
                    continue

                iou = calculate_iou(
                    [true_row['xmin'], true_row['ymin'], true_row['xmax'], true_row['ymax']],
                    [pred_row['xmin'], pred_row['ymin'], pred_row['xmax'], pred_row['ymax']],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_pred_idx is not None:
                new_true_labels.append(true_row['class'])
                new_pred_labels.append(df_pred.loc[best_pred_idx, 'class'])
                used_pred.add(best_pred_idx)
            else:
                new_true_labels.append(true_row['class'])
                new_pred_labels.append(0)

        for pred_idx, pred_row in pred_boxes.iterrows():
            if pred_idx not in used_pred:
                new_true_labels.append(0)
                new_pred_labels.append(pred_row['class'])

    conf_matrix = confusion_matrix(new_true_labels, new_pred_labels, labels=[0, 1, 2, 3, 4])
    return conf_matrix, new_true_labels, new_pred_labels
