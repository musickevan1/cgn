import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.connectome_analysis.evaluation.statistical_tests import compare_models_paired_t_test, permutation_test, bootstrap_confidence_interval

def compare_models_report(
    model_results: Dict[str, Dict[str, Any]],
    metric: str = 'test_auroc_mean',
    baseline_model_name: Optional[str] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates a report comparing multiple models based on a specified metric,
    including statistical tests against a baseline model if provided.

    Args:
        model_results (Dict[str, Dict[str, Any]]): A dictionary where keys are model names
                                                   and values are dictionaries of aggregated
                                                   cross-validation results (e.g., from CrossValidator).
                                                   Each result dict should contain '{metric}_mean' and '{metric}_std'.
        metric (str): The metric to use for comparison (e.g., 'test_auroc_mean').
        baseline_model_name (Optional[str]): The name of the baseline model to compare against.
                                              If None, no statistical comparison is performed.
        n_permutations (int): Number of permutations for permutation test.
        random_state (Optional[int]): Random state for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame summarizing model performance and statistical comparisons.
    """
    report_data = []
    
    # Prepare data for statistical tests if a baseline is provided
    baseline_scores = None
    if baseline_model_name and baseline_model_name in model_results:
        # Assuming model_results contains individual fold scores for statistical tests
        # This requires a change in CrossValidator to return individual fold metrics, not just aggregated
        # For now, we'll assume the 'overall_metrics' or similar contains the list of scores per fold
        # Or, we need to adjust the input structure.
        # Let's assume for now that model_results[model_name]['fold_metrics'] is a list of dicts
        # and we can extract the metric from each fold.
        
        # Placeholder: In a real scenario, CrossValidator should return a list of scores per fold
        # For now, we'll use the mean as a single score for simplicity in this report,
        # but for proper statistical tests, we need per-fold scores.
        # If we only have mean/std, we can't do paired t-tests or permutation tests directly.
        # This function needs to be refined once the CrossValidator output is finalized.
        
        # For demonstration, let's assume 'metric' refers to the mean, and we need to get
        # the actual per-fold scores from a different key, e.g., 'all_fold_scores_for_metric'
        # in the model_results dictionary.
        
        # For now, let's just report means and stds, and add a placeholder for statistical tests.
        pass # Will implement statistical tests properly once per-fold scores are available

    for model_name, results in model_results.items():
        mean_metric = results.get(metric)
        std_metric = results.get(metric.replace('_mean', '_std')) # Assuming naming convention
        
        row = {
            "Model": model_name,
            f"{metric.replace('_mean', '')} Mean": mean_metric,
            f"{metric.replace('_mean', '')} Std": std_metric,
            "P-value vs Baseline": np.nan,
            "Effect Size vs Baseline": np.nan
        }
        report_data.append(row)

    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values(by=f"{metric.replace('_mean', '')} Mean", ascending=False).reset_index(drop=True)
    
    # Placeholder for actual statistical tests
    if baseline_model_name and baseline_model_name in model_results:
        print("\nStatistical comparison against baseline model (placeholder - requires per-fold scores):")
        # Example:
        # baseline_scores = [fold['metric_value'] for fold in model_results[baseline_model_name]['individual_fold_results']]
        # for model_name, results in model_results.items():
        #     if model_name == baseline_model_name:
        #         continue
        #     model_scores = [fold['metric_value'] for fold in results['individual_fold_results']]
        #     
        #     # Paired t-test
        #     ttest_results = compare_models_paired_t_test(np.array(model_scores), np.array(baseline_scores))
        #     print(f"  {model_name} vs {baseline_model_name} (Paired t-test): p={ttest_results['p_value']:.4f}")
        #     
        #     # Permutation test
        #     perm_results = permutation_test(np.array(model_scores), np.array(baseline_scores), n_permutations=n_permutations, random_state=random_state)
        #     print(f"  {model_name} vs {baseline_model_name} (Permutation test): p={perm_results['p_value']:.4f}")

    return report_df

def generate_full_report(
    all_model_results: Dict[str, Dict[str, Any]],
    primary_metric: str = 'test_auroc_mean',
    baseline_model_name: Optional[str] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates a comprehensive report including model comparison and aggregated metrics.

    Args:
        all_model_results (Dict[str, Dict[str, Any]]): Dictionary of all model results.
        primary_metric (str): The main metric to use for the comparison report.
        baseline_model_name (Optional[str]): Name of the baseline model for comparison.
        n_permutations (int): Number of permutations for statistical tests.
        random_state (Optional[int]): Random state for reproducibility.

    Returns:
        Dict[str, Any]: A dictionary containing the comparison report DataFrame and other aggregated data.
    """
    report = {}
    
    # Generate model comparison table
    comparison_df = compare_models_report(
        all_model_results,
        metric=primary_metric,
        baseline_model_name=baseline_model_name,
        n_permutations=n_permutations,
        random_state=random_state
    )
    report['model_comparison_table'] = comparison_df
    
    # Add aggregated metrics for all models
    report['aggregated_metrics'] = all_model_results
    
    # Placeholder for other report components (e.g., plots, detailed per-fold results)
    
    return report
