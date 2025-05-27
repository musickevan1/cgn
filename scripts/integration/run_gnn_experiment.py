import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import yaml

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import custom models and training/evaluation components
from src.connectome_analysis.models.gnn_models import create_gnn_model # Use the factory function
from src.connectome_analysis.training.trainer import CrossValidator, ConnectomeDataset
from src.connectome_analysis.evaluation.metrics import calculate_classification_metrics, plot_roc_curve
from src.connectome_analysis.evaluation.reporting import generate_results_report
from src.connectome_analysis.evaluation.model_comparison import perform_multiple_model_comparison
# Removed interpret_gat_attention as it is not defined in attention_analysis.py

def build_gnn_model(model_name: str, config: Dict[str, Any]):
    """Helper function to build a GNN model instance using the factory."""
    return create_gnn_model(model_name, config)

def run_gnn_experiment(data_path: str, output_dir: str, config_path: Optional[str]):
    """End-to-end training and evaluation of GNN classifiers using CrossValidator."""
    print(f"Loading processed data from {data_path}...")
    try:
        with open(os.path.join(data_path, "graph_data.pkl"), "rb") as f:
            graph_data = pickle.load(f)
        with open(os.path.join(data_path, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)
        with open(os.path.join(data_path, "sites.pkl"), "rb") as f:
            sites = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data files not found in {data_path}. Please run prepare_abide_data.py first.")
        return
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return

    print(f"Loaded graph data: {len(graph_data)} graphs.")
    print(f"Loaded labels shape: {len(labels)}")
    print(f"Loaded sites shape: {len(sites)}")

    # Create a ConnectomeDataset instance
    dataset = ConnectomeDataset(graph_data, labels, sites, model_type='gnn')
    print(f"Created dataset with {len(dataset)} subjects.")

    # Load experiment configuration
    if config_path and os.path.exists(config_path):
        print(f"Loading experiment configuration from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading experiment config from {config_path}: {e}. Using placeholder config.")
            experiment_config = {
                'cv_folds': 5,
                'stratify_by_site': True,
                'random_state': 42,
                'model_type': 'gnn',
                'batch_size': 32,
                'max_epochs': 100,
                'accelerator': 'auto',
                'devices': 'auto',
                'models': {
                    'GCN': {
                        'model_name': 'GCN',
                        'model_params': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2},
                        'trainer_params': {}
                    },
                    'GAT': {
                        'model_name': 'GAT',
                        'model_params': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2, 'heads': 4},
                        'trainer_params': {}
                    },
                    'GIN': {
                        'model_name': 'GIN',
                        'model_params': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2},
                        'trainer_params': {}
                    }
                },
                'evaluation': {
                    'alpha': 0.05,
                    'correction_method': 'fdr'
                }
            }
    else:
        print("No experiment config path provided or file not found. Using placeholder configuration.")
        experiment_config = {
            'cv_folds': 5,
            'stratify_by_site': True,
            'random_state': 42,
            'model_type': 'gnn',
            'batch_size': 32,
            'max_epochs': 100,
            'accelerator': 'auto',
            'devices': 'auto',
            'models': {
                'GCN': {
                    'model_name': 'GCN',
                    'model_params': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2},
                    'trainer_params': {}
                },
                'GAT': {
                    'model_name': 'GAT',
                    'model_params': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2, 'heads': 4},
                    'trainer_params': {}
                },
                'GIN': { # GIN is not implemented in gnn_models.py, removing for now
                    'model_name': 'GIN',
                    'model_params': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2},
                    'trainer_params': {}
                }
            },
            'evaluation': {
                'alpha': 0.05,
                'correction_method': 'fdr'
            }
        }

    gnn_models_to_run = {
        "GCN": "BrainGCN", # Use string names for the factory function
        "GAT": "BrainGAT",
        # "GIN": "GIN" # GIN is not implemented in gnn_models.py
    }

    all_gnn_results: Dict[str, List[Dict[str, Any]]] = {}

    for model_name_key, model_name_str in gnn_models_to_run.items(): # Iterate over model keys and string names
        print(f"\nRunning experiment for {model_name_key}...")

        model_config = experiment_config['models'].get(model_name_key, {}) # Use model_name_key for config lookup
        if not model_config:
            print(f"Warning: Configuration for {model_name_key} not found. Skipping.")
            continue

        # The model_builder now uses the create_gnn_model factory
        def current_model_builder(params_config):
            # params_config will contain the model_params from the experiment config
            # We need to pass the model_name_str (e.g., "BrainGCN") and the actual model parameters
            # The create_gnn_model expects a config dictionary with all necessary parameters
            # So, we merge model_config['model_params']['params'] with other necessary parameters
            # like num_classes, which might be in the main experiment_config or derived.
            
            # For now, let's assume model_config['model_params']['params'] contains all needed params
            # and num_classes is available in experiment_config.
            model_params_for_factory = model_config['model_params']['params'].copy()
            model_params_for_factory['num_classes'] = experiment_config.get('num_classes', 2) # Assuming binary classification

            return create_gnn_model(model_name_str, model_params_for_factory)

        cv_config = experiment_config.copy()
        cv_config['model_type'] = 'gnn'
        cv_config['model_name'] = model_name_key # Use the key for logging/identification
        cv_config['model_params'] = model_config.get('model_params', {})
        cv_config['trainer_params'] = model_config.get('trainer_params', {})

        cross_validator = CrossValidator(
            model_builder=current_model_builder,
            dataset=dataset,
            config=cv_config
        )

        model_results = cross_validator.run_cross_validation()
        all_gnn_results.update(model_results)

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating Statistical Comparison...")
    formatted_results_for_comparison: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, fold_results_list in all_gnn_results.items():
        formatted_fold_metrics = []
        for fold_result_dict in fold_results_list:
            if fold_result_dict and isinstance(fold_result_dict, dict):
                fold_name = list(fold_result_dict.keys())[0]
                metrics = fold_result_dict[fold_name]
                if 'auc' in metrics and not np.isnan(metrics['auc']):
                    formatted_fold_metrics.append({'auc': metrics['auc']})
        if formatted_fold_metrics:
            formatted_results_for_comparison[model_name] = formatted_fold_metrics

    stat_comparison_results = None
    if len(formatted_results_for_comparison) >= 2:
        try:
            eval_config = experiment_config.get('evaluation', {})
            stat_comparison_results = perform_multiple_model_comparison(
                formatted_results_for_comparison,
                metric='auc',
                correction_method=eval_config.get('correction_method', 'fdr')
            )
            print("\nStatistical Comparison Results:")
            print(stat_comparison_results)

        except Exception as e:
            print(f"Error during statistical comparison: {e}")
            stat_comparison_results = {'error': f"Error during statistical comparison: {e}"}
    else:
        print("\nNot enough models with valid AUC scores for statistical comparison.")
        stat_comparison_results = {'message': "Not enough models for statistical comparison."}

    try:
        generate_results_report(all_gnn_results, output_dir)
        print(f"\nComprehensive results report generated in: {output_dir}")
    except Exception as e:
        print(f"\nError generating comprehensive results report: {e}")

    print("\n--- GNN Experiment Summary ---")
    for name, results_list in all_gnn_results.items():
        if results_list and "error" not in results_list[0]:
            metrics_list = []
            for fold_result_dict in results_list:
                if fold_result_dict and isinstance(fold_result_dict, dict):
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

    print("\nStatistical Comparison Summary:")
    if isinstance(stat_comparison_results, pd.DataFrame) and not stat_comparison_results.empty:
        print(stat_comparison_results)
    elif isinstance(stat_comparison_results, dict) and stat_comparison_results.get('error'):
        print(f"Statistical comparison failed: {stat_comparison_results['error']}")
    else:
        print("Statistical comparison was not performed.")

    print("\n--- GNN Experiment Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN classifier experiments on ABIDE data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing processed data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--config_path", type=str, help="Path to the experiment configuration file (optional).")

    args = parser.parse_args()
    run_gnn_experiment(args.data_path, args.output_dir, args.config_path)
