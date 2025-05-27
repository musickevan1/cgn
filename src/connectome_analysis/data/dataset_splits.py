import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Optional, Dict

def create_stratified_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified K-fold cross-validation splits.

    Args:
        labels (np.ndarray): Array of labels for stratification.
        n_splits (int): Number of folds.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Random state for reproducibility.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train_indices, val_indices) tuples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        splits.append((train_idx, val_idx))
    return splits

def create_leave_one_site_out_splits(
    site_labels: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates Leave-One-Site-Out cross-validation splits.

    Args:
        site_labels (np.ndarray): Array of site labels for each subject.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train_indices, val_indices) tuples.
    """
    unique_sites = np.unique(site_labels)
    splits = []
    for held_out_site in unique_sites:
        val_indices = np.where(site_labels == held_out_site)[0]
        train_indices = np.where(site_labels != held_out_site)[0]
        splits.append((train_indices, val_indices))
    return splits

def get_data_split_indices(
    labels: np.ndarray,
    site_labels: Optional[np.ndarray] = None,
    split_type: str = "stratified_kfold",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates data split indices based on the specified split type.

    Args:
        labels (np.ndarray): Array of labels for stratification.
        site_labels (np.ndarray, optional): Array of site labels for Leave-One-Site-Out.
        split_type (str): Type of split ('stratified_kfold' or 'leave_one_site_out').
        n_splits (int): Number of folds for stratified K-fold.
        shuffle (bool): Whether to shuffle for stratified K-fold.
        random_state (int): Random state for reproducibility.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train_indices, val_indices) tuples.

    Raises:
        ValueError: If an unsupported split type is provided or site_labels are missing for LOSO.
    """
    if split_type == "stratified_kfold":
        return create_stratified_kfold_splits(labels, n_splits, shuffle, random_state)
    elif split_type == "leave_one_site_out":
        if site_labels is None:
            raise ValueError("site_labels must be provided for 'leave_one_site_out' split type.")
        return create_leave_one_site_out_splits(site_labels)
    else:
        raise ValueError(f"Unsupported split type: {split_type}")
