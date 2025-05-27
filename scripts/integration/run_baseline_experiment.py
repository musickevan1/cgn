import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import yaml
import pickle # Added import

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import custom models and training/evaluation components
from src.connectome_analysis.data.data_modules import ConnectomeDataModule
from src.connectome_analysis.evaluation.cross_validation import CrossValidator
from src.connectome_analysis.models.baseline_models import SVMClassifier, RandomForest, LogisticRegressionClassifier
from src.connectome_analysis.evaluation.metrics import calculate_classification_metrics
from src.connectome_analysis.evaluation.reporting import generate_results_report
from src.connectome_analysis.evaluation.model_comparison import compare_models_report # Changed import

def build_baseline_model(model_name: str, model_params: Dict[str, Any]):
    """Builds a baseline model instance based on name and parameters."""
    if model_name == 'SVM':
        return SVMClassifier(**model_params)
    elif model_name == 'RandomForest':
        return RandomForest(**model_params)
    elif model_name == 'LogisticRegression':
        return LogisticRegressionClassifier(**model_params)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

def load_processed_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Loads preprocessed features, labels, and site labels from the specified path."""
    features_path = os.path.join(data_path, 'baseline_features.pkl') # Changed to baseline_features.pkl
    labels_path = os.path.join(data_path, 'labels.pkl') # Changed to labels.pkl
    site_labels_path = os.path.join(data_path, 'sites.pkl') # Changed to sites.pkl

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Processed data (baseline_features.pkl or labels.pkl) not found in {data_path}")

    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    site_labels = None
    if os.path.exists(site_labels_path):
        with open(site_labels_path, 'rb') as f:
            site_labels = pickle.load(f)
        print(f"Loaded site labels from {site_labels_path}")
    else:
        print(f"No site labels found at {site_labels_path}. Proceeding without site stratification.")

    # Ensure they are numpy arrays
    features = np.asarray(features)
    labels = np.asarray(labels)
    if site_labels is not None:
        site_labels = np.asarray(site_labels)

    print(f"Loaded features with shape: {features.shape}")
    print(f"Loaded labels with shape: {labels.shape}")
    return features, labels, site_labels

def run_baseline_experiment(data_path: str, output_dir: str, config_path: Optional[str]):
    """End-to-end training and evaluation of baseline classifiers using CrossValidator."""
    print(f"Loading processed data from {data_path}...")
    features, labels, site_labels = load_processed_data(data_path)

    # ConnectomeDataModule is now initialized with features and labels directly
    # It handles its own setup and splitting internally based on fold_idx
    # We will initialize it inside the CrossValidator loop for each fold.
    print(f"Data loaded. Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Load experiment configuration
    if config_path and os.path.exists(config_path):
        print(f"Loading experiment configuration from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading experiment config from {config_path}: {e}. Using placeholder config.")
            # Fallback to placeholder config if loading fails
            experiment_config = {
                'cv_folds': 5,
                'stratify_by_site': True,
                'random_state': 42,
                'model_type': 'baseline',
                'batch_size': 32,
                'max_epochs': 100,
                'accelerator': 'auto',
                'devices': 'auto', # Added devices config
                'models': {
                    'SVM': {
                        'model_name': 'SVM',
                        'model_params': {'params': {'C': 1.0, 'kernel': 'linear'}},
                        'trainer_params': {}
                    },
                    'RandomForest': {
                        'model_name': 'RandomForest',
                        'model_params': {'params': {'n_estimators': 100, 'max_depth': None}},
                        'trainer_params': {}
                    },
                    'LogisticRegression': {
                        'model_name': 'LogisticRegression',
                        'model_params': {'params': {'C': 1.0, 'solver': 'liblinear'}},
                        'trainer_params': {}
                    }
                },
                'evaluation': { # Added evaluation config
                    'alpha': 0.05,
                    'test_type': 'wilcoxon'
                }
            }
    else:
        print("No experiment config path provided or file not found. Using placeholder configuration.")
        # Placeholder experiment config
        experiment_config = {
            'cv_folds': 5,
            'stratify_by_site': True,
            'random_state': 42,
            'model_type': 'baseline',
            'batch_size': 32,
            'max_epochs': 100,
            'accelerator': 'auto',
            'devices': 'auto', # Added devices config
            'models': {
                'SVM': {
                    'model_name': 'SVM',
                    'model_params': {'params': {'C': 1.0, 'kernel': 'linear'}},
                    'trainer_params': {}
                },
                'RandomForest': {
                    'model_name': 'RandomForest',
                    'model_params': {'params': {'n_estimators': 100, 'max_depth': None}},
                    'trainer_params': {}
                },
                'LogisticRegression': {
                    'model_name': 'LogisticRegression',
                    'model_params': {'params': {'C': 1.0, 'solver': 'liblinear'}},
                    'trainer_params': {}
                }
            },
            'evaluation': { # Added evaluation config
                'alpha': 0.05,
                'test_type': 'wilcoxon'
            }
        }


    all_baseline_results: Dict[str, List[Dict[str, Any]]] = {} # Explicitly type

    models_to_run = {}
    if 'models' in experiment_config: # Check for the plural 'models' key
        models_to_run = experiment_config['models']
    elif 'model' in experiment_config: # Check for the singular 'model' key
        single_model_spec = experiment_config['model']
        # Try to infer a model_key and structure it like the 'models' dict
        model_name_from_params = single_model_spec.get('params', {}).get('name', 'default_model_key')
        # Reconstruct model_spec to match what the loop expects
        reconstructed_spec = {
            'model_name': model_name_from_params,
            'model_params': {'params': {k: v for k, v in single_model_spec.get('params', {}).items() if k != 'name'}}, # Pass other params
            'trainer_params': single_model_spec.get('trainer_params', {}) # Assuming trainer_params might exist
        }
        if 'type' in single_model_spec: # Add type if present
             reconstructed_spec['model_type'] = single_model_spec['type']

        models_to_run = {model_name_from_params: reconstructed_spec}
    else:
        print("Warning: Neither 'models' nor 'model' key found in experiment_config. No models to run.")

    for model_key, model_spec in models_to_run.items():
        model_name = model_spec.get('model_name')
        if not model_name:
            print(f"Warning: Model specification for key '{model_key}' is missing 'model_name'. Skipping.")
            continue
        
        # Ensure model_params is a dictionary, specifically the 'params' sub-dictionary
        # The build_baseline_model expects model_params to be the inner dict of actual parameters
        actual_model_params = model_spec.get('model_params', {}).get('params', {})


        print(f"\nRunning experiment for {model_name}...")

        # Create a model builder function for the current model
        def current_model_builder(name=model_name, params=actual_model_params):
             # The CrossValidator expects a callable that returns a model instance.
             # We use build_baseline_model to create the specific model.
             return build_baseline_model(name, params)

        # Update experiment config with model-specific details for CrossValidator
        cv_config = experiment_config.copy() # Start with the global config
        # cv_config specific to this model run
        cv_config['model_type'] = model_spec.get('model_type', 'baseline') # Get type from spec or default
        cv_config['model_name'] = model_name
        cv_config['model_params'] = model_spec.get('model_params', {}) # This should include the 'params' sub-dict
        cv_config['trainer_params'] = model_spec.get('trainer_params', {}) # Pass trainer params
        cv_config['model_builder'] = current_model_builder # Pass the model builder function
        cv_config['data_module_params'] = { # Parameters for ConnectomeDataModule
            'batch_size': experiment_config.get('batch_size', 32),
            'random_state': experiment_config.get('random_state', 42),
            'split_type': 'stratified_kfold', # Default split type for baseline
            'n_splits': experiment_config.get('cv_folds', 5),
            'site_labels': site_labels if experiment_config.get('stratify_by_site', False) else None
        }

        # Initialize CrossValidator with the config
        cross_validator = CrossValidator(config=cv_config)

        # Run cross-validation
        model_results = cross_validator.run_cross_validation(
            features=features,
            labels=labels,
            site_labels=site_labels if experiment_config.get('stratify_by_site', False) else None
        )

        # Store results in the overall dictionary
        all_baseline_results[model_name] = model_results.get('all_fold_results', [])


    # Generate comprehensive results report
    os.makedirs(output_dir, exist_ok=True)

    # 4. Generate statistical comparisons
    print("\nGenerating Statistical Comparison...")
    # Prepare data for compare_models_report
    # It expects Dict[str, Dict[str, Any]] where inner dict has '{metric}_mean' and '{metric}_std'
    aggregated_metrics_for_comparison: Dict[str, Dict[str, Any]] = {}
    for model_name, results_list in all_baseline_results.items():
        if results_list:
            # Extract AUCs from each fold result
            aucs_values = [
                fold_result.get('test_auroc_mean')
                for fold_result in results_list
                if fold_result and 'test_auroc_mean' in fold_result and not np.isnan(fold_result['test_auroc_mean'])
            ]
            if aucs_values:
                aucs_array = np.array(aucs_values, dtype=float)
                aggregated_metrics_for_comparison[model_name] = {
                    'test_auroc_mean': np.mean(aucs_array),
                    'test_auroc_std': np.std(aucs_array)
                }
            else:
                # Handle case where no valid AUCs are found for a model
                aggregated_metrics_for_comparison[model_name] = {
                    'test_auroc_mean': np.nan,
                    'test_auroc_std': np.nan
                }

    stat_comparison_results = None
    if len(aggregated_metrics_for_comparison) >= 2:
        try:
            # Use evaluation config for statistical test parameters
            eval_config = experiment_config.get('evaluation', {})
            stat_comparison_results = compare_models_report( # Changed function call
                aggregated_metrics_for_comparison,
                metric='test_auroc_mean', # Use AUC for comparison
                baseline_model_name=list(aggregated_metrics_for_comparison.keys())[0], # Use first model as baseline for report
                n_permutations=eval_config.get('n_permutations', 1000),
                random_state=eval_config.get('random_state', 42)
            )
            print("\nStatistical Comparison Results:")
            print(stat_comparison_results)

        except Exception as e:
            print(f"Error during statistical comparison: {e}")
            stat_comparison_results = {'error': f"Error during statistical comparison: {e}"}
    else:
        print("\nNot enough models with valid AUC scores for statistical comparison.")
        stat_comparison_results = {'message': "Not enough models for statistical comparison."}


    # 5. Save results in organized format & 6. Create summary report
    # The generate_results_report function handles saving and includes summary/stats/plots
    try:
         # Pass statistical comparison results to the report generator
         generate_results_report(all_baseline_results, output_dir) # Report generator calls statistical_significance_test internally
         print(f"\nComprehensive results report generated in: {output_dir}")
    except Exception as e:
         print(f"\nError generating comprehensive results report: {e}")


    print("\n--- Baseline Experiment Summary ---")
    for name, results_list in all_baseline_results.items(): # Iterate over list of fold results
        if results_list and "error" not in results_list[0]:
            # Extract metrics into a list of dicts
            metrics_list = []
            for fold_result_dict in results_list:
                 if fold_result_dict and isinstance(fold_result_dict, dict):
                      # Assuming the inner dict has only one key (the fold name)
                      fold_name = list(fold_result_dict.keys())[0]
                      metrics_list.append(fold_result_dict[fold_name])

            if metrics_list:
                 df_results = pd.DataFrame(metrics_list)
                 print(f"\n{name} (Mean CV Metrics):")
                 for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
                      if metric in df_results.columns:
                           mean = df_results[metric].mean()
                           std = df_results[metric].std()
                           print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
            else:
                 print(f"\nNo valid metrics found for {name}.")
        elif results_list and "error" in results_list[0]:
             print(f"\n{name}: Error during experiment - {results_list[0]['error']}")
        else:
             print(f"\n{name}: No results obtained.")

    # Print statistical comparison summary again for clarity in console output
    print("\nStatistical Comparison Summary:")
    if isinstance(stat_comparison_results, pd.DataFrame) and not stat_comparison_results.empty:
        print(stat_comparison_results)
    elif isinstance(stat_comparison_results, dict) and stat_comparison_results.get('error'):
         print(f"Statistical comparison failed: {stat_comparison_results['error']}")
    else:
         print("Statistical comparison was not performed.")


    print("\n--- Baseline Experiment Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline classifier experiments on ABIDE data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing processed data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--config_path", type=str, help="Path to the experiment configuration file (optional).") # Make config optional

    args = parser.parse_args()

    # If config_path is provided, load it. Otherwise, use the placeholder config.
    # The run_baseline_experiment function now handles loading the config internally
    # or using the placeholder if config_path is None.
    run_baseline_experiment(args.data_path, args.output_dir, args.config_path)
