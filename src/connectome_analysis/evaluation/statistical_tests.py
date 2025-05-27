import numpy as np
from numpy.typing import NDArray # Added
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional, Callable, cast
from sklearn.utils import resample

def permutation_test(
    group1_scores: np.ndarray,
    group2_scores: np.ndarray,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Performs a permutation test to compare two groups of scores.

    Args:
        group1_scores (np.ndarray): Scores for group 1.
        group2_scores (np.ndarray): Scores for group 2.
        n_permutations (int): Number of permutations to perform.
        random_state (Optional[int]): Random state for reproducibility.

    Returns:
        Dict[str, Any]: Dictionary containing the observed difference, p-value,
                        and permutation distribution.
    """
    np.random.seed(random_state)

    combined_scores = np.concatenate((group1_scores, group2_scores))
    n1 = len(group1_scores)
    n2 = len(group2_scores)

    observed_diff = np.mean(group1_scores) - np.mean(group2_scores)
    permutation_diffs = []

    for _ in range(n_permutations):
        permuted_combined = np.random.permutation(combined_scores)
        permuted_group1 = permuted_combined[:n1]
        permuted_group2 = permuted_combined[n1:]
        permutation_diffs.append(np.mean(permuted_group1) - np.mean(permuted_group2))

    p_value = (np.sum(np.abs(permutation_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "permutation_distribution": np.array(permutation_diffs)
    }

def bootstrap_confidence_interval(
    data: NDArray, # Changed from np.ndarray
    statistic_func: Callable[[NDArray], float], # Changed from np.ndarray
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculates a bootstrap confidence interval for a given statistic.

    Args:
        data (NDArray): Input data.
        statistic_func (Callable[[NDArray], float]): Function to compute the statistic (e.g., np.mean, np.median).
        n_bootstraps (int): Number of bootstrap samples.
        confidence_level (float): Confidence level for the interval (e.g., 0.95 for 95% CI).
        random_state (Optional[int]): Random state for reproducibility.

    Returns:
        Dict[str, Any]: Dictionary containing the observed statistic, bootstrap samples,
                        and the confidence interval.
    """
    if not isinstance(data, np.ndarray): # Keep isinstance check for runtime safety
        raise TypeError(f"Input 'data' must be a numpy.ndarray. Got {type(data)}")

    # Set random state for reproducibility
    # Create a RandomState object for resample to ensure consistent type
    if random_state is not None:
        np.random.seed(random_state) # This sets the global seed
        rng = np.random.RandomState(random_state) # This creates a local RandomState instance
    else:
        rng = None # If random_state is None, resample will use the global random state

    observed_statistic = statistic_func(data)
    bootstrap_samples: List[float] = [] # Explicitly typing bootstrap_samples

    for _ in range(n_bootstraps):
        # Pass the RandomState object or None to resample
        # Explicitly cast data to NDArray to satisfy Pylance, building on the isinstance check
        casted_data = cast(NDArray, data) # Changed from np.ndarray
        sample = cast(NDArray, resample(casted_data, replace=True, n_samples=len(casted_data), random_state=rng))
        bootstrap_samples.append(statistic_func(sample))

    alpha = 1.0 - confidence_level
    lower_bound = np.percentile(bootstrap_samples, alpha / 2 * 100)
    upper_bound = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)

    return {
        "observed_statistic": observed_statistic,
        "bootstrap_samples": np.array(bootstrap_samples),
        "confidence_interval": (lower_bound, upper_bound)
    }

def compare_models_paired_t_test(
    model1_scores: np.ndarray,
    model2_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Performs a paired t-test to compare the performance of two models
    on the same set of samples (e.g., cross-validation folds).

    Args:
        model1_scores (np.ndarray): Scores of model 1 across samples/folds.
        model2_scores (np.ndarray): Scores of model 2 across samples/folds.

    Returns:
        Dict[str, Any]: Dictionary containing t-statistic, p-value, and degrees of freedom.
    """
    if len(model1_scores) != len(model2_scores):
        raise ValueError("Scores arrays must have the same length for paired t-test.")

    t_statistic, p_value = stats.ttest_rel(model1_scores, model2_scores)
    degrees_of_freedom = len(model1_scores) - 1

    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom
    }
