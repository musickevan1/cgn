import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import zscore

def detect_outliers_zscore(
    data: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detects outliers in a 1D array using the Z-score method.

    Args:
        data (np.ndarray): 1D array of numerical data.
        threshold (float): Z-score threshold for outlier detection.

    Returns:
        np.ndarray: Boolean array indicating outliers (True for outliers).
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    
    # Handle NaNs by ignoring them in zscore calculation and marking them as non-outliers
    non_nan_data = data[~np.isnan(data)]
    if len(non_nan_data) == 0:
        return np.zeros_like(data, dtype=bool)
    
    z_scores = np.abs(zscore(non_nan_data))
    outliers_non_nan = z_scores > threshold
    
    outliers = np.zeros_like(data, dtype=bool)
    outliers[~np.isnan(data)] = outliers_non_nan
    
    return outliers

def detect_outliers_iqr(
    data: np.ndarray,
    iqr_multiplier: float = 1.5
) -> np.ndarray:
    """
    Detects outliers in a 1D array using the Interquartile Range (IQR) method.

    Args:
        data (np.ndarray): 1D array of numerical data.
        iqr_multiplier (float): Multiplier for the IQR to define outlier bounds.

    Returns:
        np.ndarray: Boolean array indicating outliers (True for outliers).
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    q1, q3 = np.percentile(data[~np.isnan(data)], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * iqr_multiplier)
    upper_bound = q3 + (iqr * iqr_multiplier)

    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

def perform_quality_control(
    phenotypic_data: pd.DataFrame,
    qc_metrics: Dict[str, Dict[str, Any]],
    outlier_detection_method: str = "zscore"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs quality control on phenotypic data based on specified metrics and outlier detection.

    Args:
        phenotypic_data (pd.DataFrame): DataFrame containing phenotypic data and QC metrics.
        qc_metrics (Dict[str, Dict[str, Any]]): Dictionary defining QC metrics and their thresholds.
            Example:
            {
                "mean_fd": {"threshold": 0.2, "method": "zscore"},
                "t_compcor_ncomponents": {"threshold": 5, "method": "iqr", "direction": "lower_bound"}
            }
        outlier_detection_method (str): Default outlier detection method ('zscore' or 'iqr').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - qc_passed_data (pd.DataFrame): DataFrame with subjects passing QC.
            - qc_failed_report (pd.DataFrame): DataFrame reporting subjects failing QC.
    """
    qc_failed_subjects = []
    qc_passed_data = phenotypic_data.copy()
    
    for metric, params in qc_metrics.items():
        if metric not in phenotypic_data.columns:
            print(f"Warning: QC metric '{metric}' not found in phenotypic data. Skipping.")
            continue

        data_to_check = phenotypic_data[metric].values.astype(float) # Ensure it's a numpy array of floats
        method = params.get("method", outlier_detection_method)
        threshold = params.get("threshold")
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Threshold for metric '{metric}' must be a number, got {type(threshold)}")
        direction = params.get("direction", "both") # 'both', 'upper_bound', 'lower_bound'

        if method == "zscore":
            is_outlier = detect_outliers_zscore(data_to_check, threshold=float(threshold))
        elif method == "iqr":
            is_outlier = detect_outliers_iqr(data_to_check, iqr_multiplier=float(threshold)) # Using threshold as iqr_multiplier
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        # Refine outlier detection based on direction
        if direction == "upper_bound":
            is_outlier = is_outlier & (data_to_check > np.mean(data_to_check[~np.isnan(data_to_check)])) # Only consider values above mean as outliers
        elif direction == "lower_bound":
            is_outlier = is_outlier & (data_to_check < np.mean(data_to_check[~np.isnan(data_to_check)])) # Only consider values below mean as outliers

        failed_indices = np.where(is_outlier)[0]
        for idx in failed_indices:
            subject_id = phenotypic_data.iloc[idx].get("participant_id", f"Index_{idx}")
            qc_failed_subjects.append({
                "participant_id": subject_id,
                "metric": metric,
                "value": phenotypic_data.iloc[idx][metric],
                "threshold": threshold,
                "method": method,
                "reason": f"Outlier detected by {method} method for {metric}"
            })
        
        # Remove failed subjects from qc_passed_data
        qc_passed_data = qc_passed_data[~is_outlier]

    qc_failed_report = pd.DataFrame(qc_failed_subjects)
    return qc_passed_data, qc_failed_report
