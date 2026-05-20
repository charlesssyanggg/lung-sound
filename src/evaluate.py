"""
evaluate.py — Evaluation Metrics

Implements:
  - ICBHI Score (official challenge metric)
  - Per-class Sensitivity & Specificity
  - Macro F1, Accuracy
  - Confusion matrix
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, recall_score,
    f1_score, accuracy_score
)

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


def icbhi_score(y_true: np.ndarray,
                y_pred: np.ndarray) -> tuple:
    """
    Compute the official ICBHI 2017 challenge score.

    Score = (macro-Sensitivity + macro-Specificity) / 2

    Sensitivity_i = TP_i / (TP_i + FN_i)   (per-class recall)
    Specificity_i = TN_i / (TN_i + FP_i)   (per-class)
    Macro-SE = mean(SE_i);  Macro-SP = mean(SP_i)

    Args:
        y_true: ground-truth labels (N,)
        y_pred: predicted labels    (N,)

    Returns:
        (score, macro_se, macro_sp)   all in [0, 1]
    """
    n_cls = len(CLASS_NAMES)
    se_list, sp_list = [], []
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))

    for i in range(n_cls):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        se = tp / (tp + fn + 1e-9)
        sp = tn / (tn + fp + 1e-9)
        se_list.append(se)
        sp_list.append(sp)

    macro_se = np.mean(se_list)
    macro_sp = np.mean(sp_list)
    score    = (macro_se + macro_sp) / 2.0
    return score, macro_se, macro_sp


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: str = "cpu") -> dict:
    """
    Evaluate model on a DataLoader and return a metrics dict.

    Args:
        model:  PyTorch model in eval mode.
        loader: DataLoader yielding (x, y) batches.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with keys:
            icbhi_score, sensitivity, specificity,
            accuracy, f1_macro,
            per_class_se (dict),
            y_true (list), y_pred (list)
    """
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu()
        all_preds.extend(pred.numpy())
        all_labels.extend(
            y.numpy() if isinstance(y, torch.Tensor) else y
        )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    score, macro_se, macro_sp = icbhi_score(y_true, y_pred)

    per_class_se = recall_score(
        y_true, y_pred,
        average=None, zero_division=0,
        labels=list(range(len(CLASS_NAMES)))
    )

    return {
        "icbhi_score":  round(score * 100, 2),
        "sensitivity":  round(macro_se * 100, 2),
        "specificity":  round(macro_sp * 100, 2),
        "accuracy":     round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_macro":     round(
            f1_score(y_true, y_pred,
                     average="macro", zero_division=0) * 100, 2
        ),
        "per_class_se": {
            CLASS_NAMES[i]: round(per_class_se[i] * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def print_metrics(metrics: dict, title: str = "Results"):
    """Pretty-print evaluation metrics."""
    line = "─" * 50
    print(f"\n{line}")
    print(f"  {title}")
    print(line)
    print(f"  ICBHI Score  : {metrics['icbhi_score']:.2f}%")
    print(f"  Sensitivity  : {metrics['sensitivity']:.2f}%")
    print(f"  Specificity  : {metrics['specificity']:.2f}%")
    print(f"  Accuracy     : {metrics['accuracy']:.2f}%")
    print(f"  F1 (Macro)   : {metrics['f1_macro']:.2f}%")
    print(f"  Per-class SE :")
    for name, se in metrics["per_class_se"].items():
        bar = "█" * int(se / 5)
        print(f"    {name:8s}: {se:5.1f}%  {bar}")
    print(line)
