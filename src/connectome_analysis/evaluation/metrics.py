import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)
from typing import Dict, Any, Optional, Union, Literal

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: Literal['binary', 'micro', 'macro', 'weighted', None] = 'binary'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculates a comprehensive set of classification metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class (binary)
                                       or for each class (multiclass). Required for AUC.
        average (str): Averaging strategy for metrics like precision, recall, f1-score.

    Returns:
        Dict[str, Union[float, np.ndarray]]: A dictionary of calculated metrics.
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # AUC-ROC
    if y_prob is not None:
        try:
            # For binary classification, y_prob should be 1D (probabilities of the positive class)
            # For multiclass, y_prob should be 2D (probabilities for each class)
            if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 2): # Binary or 2-class multiclass
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            else: # Multiclass with more than 2 classes
                # For multiclass AUC, 'average' parameter for roc_auc_score does not accept 'binary'
                # It expects 'micro', 'macro', 'weighted', or None.
                # If 'average' is 'binary', we should use 'macro' or 'weighted' for multiclass AUC.
                # Let's default to 'macro' for multiclass AUC if 'binary' was passed.
                auc_average = average if average != 'binary' else 'macro'
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=auc_average)
        except ValueError as e:
            print(f"Warning: Could not calculate AUC-ROC. Error: {e}")
            metrics['auc_roc'] = np.nan
    else:
        metrics['auc_roc'] = np.nan

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics

# Placeholder for other potential metrics or metric-related utilities
# def plot_roc_curve(...):
#     pass

# def plot_confusion_matrix(...):
#     pass
