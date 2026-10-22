"""
Model Evaluation and Metric Calculation Module for Social Media Sentiment Analysis.

Calculates:
- Accuracy, Precision, Recall, F1-Score (macro, weighted, per-class)
- Confusion Matrix
- ROC-AUC Score (when prediction probabilities are provided)
- Human-readable and JSON-serializable evaluation summaries
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    y_proba: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for model evaluation.

    Args:
        y_true: True ground truth labels (0=negative, 1=positive).
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class (or 2D matrix).
        target_names: Names of classes, default ['negative', 'positive'].

    Returns:
        Dictionary containing all metrics in native serializable Python types.
    """
    if target_names is None:
        target_names = ["negative", "positive"]

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    # Per-class metrics
    report = classification_report(
        y_true_arr, y_pred_arr, target_names=target_names, output_dict=True, zero_division=0
    )

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "classes": target_names,
        "classification_report": report,
    }

    # Calculate ROC-AUC if probabilities are supplied
    if y_proba is not None:
        try:
            if hasattr(y_proba, "ndim") and y_proba.ndim == 2:
                prob_pos = y_proba[:, 1]
            else:
                prob_pos = y_proba
            auc = float(roc_auc_score(y_true_arr, prob_pos))
            metrics["roc_auc"] = round(auc, 4)
        except Exception:
            metrics["roc_auc"] = None

    return metrics
