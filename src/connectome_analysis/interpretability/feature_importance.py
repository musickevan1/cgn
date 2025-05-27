import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Added import
from typing import Dict, Any, List, Tuple, Optional, cast # Added cast
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch # Import Bunch for type hinting

def calculate_permutation_feature_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = 'accuracy',
    n_repeats: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculates feature importance using permutation importance for classical ML models.

    Args:
        model (BaseEstimator): Trained scikit-learn compatible model.
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        scoring (str): Scoring metric to use (e.g., 'accuracy', 'roc_auc').
        n_repeats (int): Number of times to permute a feature.
        random_state (Optional[int]): Random state for reproducibility.

    Returns:
        Dict[str, Any]: Dictionary containing feature importances (mean and std).
    """
    # permutation_importance can return dict[str, Bunch] if scoring is a dict of scorers.
    # Since 'scoring' is explicitly a single string in this function's signature,
    # the result is expected to be a single Bunch object. We use cast to inform Pylance.
    result: Bunch = cast(Bunch, permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1 # Use all available CPU cores
    )) # Added closing parenthesis for cast

    return {
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std,
        "feature_names": getattr(X, 'columns', [f'feature_{i}' for i in range(X.shape[1])]) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])] # Handle DataFrame or NumPy
    }

def visualize_feature_importance(
    importances_mean: np.ndarray,
    importances_std: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6),
    top_n: Optional[int] = None
):
    """
    Visualizes feature importances using a bar plot.

    Args:
        importances_mean (np.ndarray): Mean importance scores for each feature.
        importances_std (np.ndarray): Standard deviation of importance scores.
        feature_names (List[str]): Names of the features.
        title (str): Title of the plot.
        figsize (Tuple[int, int]): Figure size.
        top_n (Optional[int]): If provided, shows only the top N most important features.
    """
    if len(importances_mean) != len(feature_names):
        raise ValueError("Length of importances_mean and feature_names must match.")

    # Create a DataFrame for easy sorting and plotting
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importances_mean,
        'importance_std': importances_std
    })
    df = df.sort_values(by='importance_mean', ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(df['feature'], df['importance_mean'], xerr=df['importance_std'], align='center')
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis() # Most important feature at the top
    plt.tight_layout()
    plt.show()
