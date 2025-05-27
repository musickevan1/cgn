import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import necessary components from the pipeline
from scripts.integration.run_baseline_experiment import run_baseline_experiment
from scripts.integration.run_gnn_experiment import run_gnn_experiment
from connectome_analysis.evaluation.metrics import calculate_classification_metrics # Import relevant metrics functions
from src.connectome_analysis.evaluation.model_comparison import perform_multiple_model_comparison
from src.connectome_analysis.evaluation.reporting import generate_results_report

def run_benchmarking(data_path, output_dir, config_path=None):
    """
    Run benchmarking experiments on ABIDE data and compare against literature.

    Args:
        data_path: Path to the directory containing processed data.
        output_dir: Directory to save benchmarking results and reports.
        config_path: Path to the experiment configuration file (optional).
    """
    print("Running benchmarking experiments...")

    # Define output directories for baseline and GNN results within the benchmark output directory
    baseline_output_dir = os.path.join(output_dir, 'baselines')
    gnn_output_dir = os.path.join(output_dir, 'gnn')
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(gnn_output_dir, exist_ok=True)

    # Run baseline experiments
    print("\n--- Running Baseline Benchmarking ---")
    # Assuming run_baseline_experiment can be called as a function
    # and handles its own configuration loading if config_path is None
    run_baseline_experiment(data_path, baseline_output_dir, config_path)
    print("Baseline benchmarking complete.")

    # Run GNN experiments
    print("\n--- Running GNN Benchmarking ---")
    # Assuming run_gnn_experiment can be called as a function
    run_gnn_experiment(data_path, gnn_output_dir, config_path)
    print("GNN benchmarking complete.")

    # Load results from the individual experiment runs
    all_results = {}
    baseline_results_path = os.path.join(baseline_output_dir, "baseline_experiment_results.pkl") # Assuming filename
    gnn_results_path = os.path.join(gnn_output_dir, "gnn_experiment_results.pkl") # Assuming filename

    if os.path.exists(baseline_results_path):
        try:
            with open(baseline_results_path, "rb") as f:
                all_results.update(pickle.load(f))
            print("\nLoaded baseline results for final report.")
        except Exception as e:
            print(f"Error loading baseline results: {e}")

    if os.path.exists(gnn_results_path):
        try:
            with open(gnn_results_path, "rb") as f:
                all_results.update(pickle.load(f))
            print("Loaded GNN results for final report.")
        except Exception as e:
            print(f"Error loading GNN results: {e}")

    # Generate a combined results report
    if all_results:
        print("\n--- Generating Combined Results Report ---")
        generate_results_report(all_results, output_dir)
        print("Combined results report generated.")
    else:
        print("\nNo results loaded to generate combined report.")


    # Perform statistical significance testing
    # Perform statistical significance testing
    if all_results and len(all_results) > 1:
        print("\n--- Performing Statistical Significance Testing ---")
        # perform_multiple_model_comparison expects Dict[str, List[Dict[str, Any]]] (model_name: [{'metric': value}, ...])
        # We have all_results: Dict[str, List[Dict[str, Any]]] (model_name: [{'fold_0':{...}}, ...])
        # Need to reformat all_results to be compatible with perform_multiple_model_comparison
        # The function expects a dictionary where values are lists of dictionaries, each containing the metric.
        # Example: {'ModelA': [{'accuracy': 0.8}, {'accuracy': 0.85}], ...}
        
        # Reformat all_results to be compatible with perform_multiple_model_comparison
        # This involves extracting the relevant metric (e.g., 'auc') from the nested structure
        # and ensuring the format is {'model_name': [{'metric_name': value}, ...]}
        formatted_results_for_comparison: Dict[str, List[Dict[str, Any]]] = {}
        for model_name, fold_results_list in all_results.items():
            formatted_fold_metrics = []
            for fold_result_dict in fold_results_list:
                if fold_result_dict and isinstance(fold_result_dict, dict):
                    # Assuming the inner dict has only one key (the fold name)
                    fold_name = list(fold_result_dict.keys())[0]
                    metrics = fold_result_dict[fold_name]
                    if 'auc' in metrics and not np.isnan(metrics['auc']):
                        formatted_fold_metrics.append({'auc': metrics['auc']})
            if formatted_fold_metrics:
                formatted_results_for_comparison[model_name] = formatted_fold_metrics

        stat_comparison_results = None
        if len(formatted_results_for_comparison) >= 2:
            try:
                # Use default parameters for statistical test if not in config
                stat_comparison_results = perform_multiple_model_comparison(
                    formatted_results_for_comparison,
                    metric='auc', # Use AUC for comparison
                    correction_method='fdr' # Default correction method
                )
                print("\nStatistical Comparison Results:")
                print(stat_comparison_results)

            except Exception as e:
                print(f"Error during statistical comparison: {e}")
                stat_comparison_results = {'error': f"Error during statistical comparison: {e}"}
        else:
            print("\nNot enough models with valid AUC scores for statistical comparison.")
            stat_comparison_results = {'message': "Not enough models for statistical comparison."}

    else:
        print("\nSkipping statistical significance testing (requires results from at least two models).")
        stat_comparison_results = {'message': "Skipping statistical significance testing."}


    # Compare against literature benchmarks
    print("\n--- Comparison to Literature Benchmarks ---")
    # This part requires hardcoded or loaded literature benchmark values
    literature_benchmarks = {
        'SVM': {'accuracy': 0.60, 'auc': 0.65}, # Example values
        'RandomForest': {'accuracy': 0.61, 'auc': 0.66},
        'LogisticRegression': {'accuracy': 0.60, 'auc': 0.64},
        'BrainGCN': {'accuracy': 0.70, 'auc': 0.75},
        'BrainGAT': {'accuracy': 0.72, 'auc': 0.77}
    }

    if all_results:
        print("Comparing mean CV metrics to literature:")
        for model_name, results_list in all_results.items(): # Iterate over list of fold results
            if results_list and "error" not in results_list[0] and model_name in literature_benchmarks:
                # Extract metrics into a list of dicts
                metrics_list = []
                for fold_result_dict in results_list:
                     if fold_result_dict and isinstance(fold_result_dict, dict):
                          # Assuming the inner dict has only one key (the fold name)
                          fold_name = list(fold_result_dict.keys())[0]
                          metrics_list.append(fold_result_dict[fold_name])

                if metrics_list:
                     df_results = pd.DataFrame(metrics_list)
                     mean_metrics = df_results.mean().to_dict()
                     print(f"  {model_name}:")
                     for metric in literature_benchmarks[model_name]:
                          if metric in mean_metrics:
                               print(f"    Mean CV {metric.capitalize()}: {mean_metrics[metric]:.4f} (Literature: {literature_benchmarks[model_name][metric]:.4f})")
            elif model_name in literature_benchmarks:
                 print(f"  {model_name}: Results not available for comparison.")
            else:
                 print(f"  {model_name}: Not in literature benchmarks or no results obtained.")

    else:
        print("No results available for comparison to literature.")


    # The combined report generated earlier includes statistical tests and plots
    # if the necessary data was available.
    print(f"\nCombined results report saved to: {os.path.join(output_dir, 'combined_experiment_report.html')}")


    print("\n--- Benchmarking Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking experiments on ABIDE data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing processed data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save benchmarking results and reports.")
    parser.add_argument("--config_path", type=str, help="Path to the experiment configuration file (optional).")

    args = parser.parse_args()

    run_benchmarking(args.data_path, args.output_dir, args.config_path)
