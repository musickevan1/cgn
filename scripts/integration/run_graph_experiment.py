import sys
import os
import argparse
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import yaml

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import custom models and training/evaluation components
from src.connectome_analysis.models import create_gnn_model, create_transformer_model # Import GNN and Transformer model factories
from src.connectome_analysis.training.trainer import ConnectomeTrainer, CrossValidator, ConnectomeDataset
from src.connectome_analysis.evaluation.metrics import calculate_classification_metrics, plot_roc_curve # Corrected imports
from src.connectome_analysis.evaluation.reporting import generate_results_report # Corrected import
from src.connectome_analysis.evaluation.model_comparison import perform_multiple_model_comparison # Corrected import
# Removed interpret_gat_attention as it is not defined

def run_graph_experiment(data_path: str, output_dir: str, config_path: Optional[str]):
    """End-to-end training and evaluation of Graph-based Models (GNNs and Transformers) using CrossValidator."""
    print(f"Loading processed graph data from {data_path}...")
    try:
        with open(os.path.join(data_path, "graph_data_list.pkl"), "rb") as f:
            graph_data_list = pickle.load(f)
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

    print(f"Loaded {len(graph_data_list)} graph data objects.")
    print(f"Loaded labels shape: {len(labels)}")
    print(f"Loaded sites shape: {len(sites)}")

    dataset = ConnectomeDataset(graph_data_list, labels, sites, model_type='gnn')
    print(f"Created dataset with {len(dataset)} subjects.")

    if config_path and os.path.exists(config_path):
        print(f"Loading experiment configuration from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading experiment config from {config_path}: {e}. Using placeholder config.")
            num_node_features = graph_data_list[0].x.shape[1] if graph_data_list and hasattr(graph_data_list[0], 'x') and graph_data_list[0].x is not None else 0
            num_classes = len(np.unique(labels)) if labels else 0

            experiment_config = {
                'cv_folds': 5,
                'stratify_by_site': True,
                'random_state': 42,
                'model_type': 'gnn', # Default model_type for placeholder
                'batch_size': 32,
                'max_epochs': 100,
                'accelerator': 'auto',
                'devices': 'auto',
                'models': {
                    'BrainGCN': {
                        'model_name': 'BrainGCN',
                        'model_params': {'in_channels': num_node_features, 'hidden_channels': 64, 'out_channels': 32, 'num_classes': num_classes},
                        'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                    },
                    'BrainGAT': {
                        'model_name': 'BrainGAT',
                        'model_params': {'in_channels': num_node_features, 'hidden_channels': 64, 'out_channels': 32, 'num_classes': num_classes, 'heads': 4},
                        'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                    },
                    'BrainGraphTransformer': {
                        'model_name': 'BrainGraphTransformer',
                        'model_params': {'in_channels': num_node_features, 'd_model': 64, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1, 'num_classes': num_classes},
                        'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                    }
                },
                'evaluation': {
                    'alpha': 0.05,
                    'test_type': 'wilcoxon'
                }
            }
    else:
        print("No experiment config path provided or file not found. Using placeholder configuration.")
        num_node_features = graph_data_list[0].x.shape[1] if graph_data_list and hasattr(graph_data_list[0], 'x') and graph_data_list[0].x is not None else 0
        num_classes = len(np.unique(labels)) if labels else 0

        experiment_config = {
            'cv_folds': 5,
            'stratify_by_site': True,
            'random_state': 42,
            'model_type': 'gnn', # Default model_type for placeholder
            'batch_size': 32,
            'max_epochs': 100,
            'accelerator': 'auto',
            'devices': 'auto',
            'models': {
                'BrainGCN': {
                    'model_name': 'BrainGCN',
                    'model_params': {'in_channels': num_node_features, 'hidden_channels': 64, 'out_channels': 32, 'num_classes': num_classes},
                    'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                },
                'BrainGAT': {
                    'model_name': 'BrainGAT',
                    'model_params': {'in_channels': num_node_features, 'hidden_channels': 64, 'out_channels': 32, 'num_classes': num_classes, 'heads': 4},
                    'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                },
                'BrainGraphTransformer': {
                    'model_name': 'BrainGraphTransformer',
                    'model_params': {'in_channels': num_node_features, 'd_model': 64, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1, 'num_classes': num_classes},
                    'trainer_params': {'optimizer': {'type': 'Adam', 'lr': 0.001}}
                }
            },
            'evaluation': {
                'alpha': 0.05,
                'test_type': 'wilcoxon'
            }
        }

    gnn_models_to_run = [
        "BrainGCN",
        "BrainGAT",
        "BrainGraphTransformer"
    ]

    all_gnn_results: Dict[str, List[Dict[str, Any]]] = {}

    for model_name in gnn_models_to_run:
        print(f"\nRunning experiment for {model_name}...")

        model_config = experiment_config['models'].get(model_name, {})
        if not model_config:
            print(f"Warning: Configuration for {model_name} not found. Skipping.")
            continue

        def current_model_builder(params_config):
            if model_name in ["BrainGCN", "BrainGAT"]:
                return create_gnn_model(model_name, params_config)
            elif model_name == "BrainGraphTransformer":
                return create_transformer_model(model_name, params_config)
            else:
                raise ValueError(f"Unknown model_name for graph experiment: {model_name}")

        cv_config = experiment_config.copy()
        if model_name in ["BrainGCN", "BrainGAT"]:
            cv_config['model_type'] = 'gnn'
        elif model_name == "BrainGraphTransformer":
            cv_config['model_type'] = 'transformer'
        else:
            raise ValueError(f"Unknown model_name for CrossValidator model_type: {model_name}")

        cv_config['model_name'] = model_name
        cv_config['model_params'] = model_config.get('model_params', {})
        cv_config['trainer_params'] = model_config.get('trainer_params', {})

        cross_validator = CrossValidator(
            model_builder=current_model_builder,
            dataset=dataset,
            config=cv_config
        )

        model_results = cross_validator.run_cross_validation()

        all_gnn_results.update(model_results)

    baseline_results = {}
    baseline_results_path = os.path.join(output_dir, "baseline_experiment_results.pkl")
    if os.path.exists(baseline_results_path):
        try:
            with open(baseline_results_path, "rb") as f:
                baseline_results = pickle.load(f)
            print("\nLoaded baseline results for comparison.")
        except Exception as e:
            print(f"Error loading baseline results: {e}")

    all_results = {**baseline_results, **all_gnn_results}

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating Statistical Comparison...")
    # perform_multiple_model_comparison expects Dict[str, List[Dict[str, Any]]] (model_name: [{'metric': value}, ...])
    # We have all_results: Dict[str, List[Dict[str, Any]]] (model_name: [{'fold_0':{...}}, ...])
    # Need to reformat all_results to be compatible with perform_multiple_model_comparison
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
            eval_config = experiment_config.get('evaluation', {})
            stat_comparison_results = perform_multiple_model_comparison(
                formatted_results_for_comparison,
                metric='auc', # Use AUC for comparison
                correction_method=eval_config.get('correction_method', 'fdr') # Default correction method
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
         generate_results_report(all_results, output_dir)
         print(f"\nComprehensive results report generated in: {output_dir}")
    except Exception as e:
         print(f"\nError generating comprehensive results report: {e}")

    print("\n--- Graph-based Model Experiment Summary ---")
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

    print("\nStatistical Comparison Summary (Graph-based Models vs Baseline):")
    if isinstance(stat_comparison_results, pd.DataFrame) and not stat_comparison_results.empty:
        print(stat_comparison_results)
    elif isinstance(stat_comparison_results, dict) and stat_comparison_results.get('error'):
         print(f"Statistical comparison failed: {stat_comparison_results['error']}")
    else:
         print("Statistical comparison was not performed.")

    print("\n--- Graph-based Model Experiment Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph-based model experiments (GNNs and Transformers) on ABIDE data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing processed data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--config_path", type=str, help="Path to the experiment configuration file (optional).")

    args = parser.parse_args()

    run_graph_experiment(args.data_path, args.output_dir, args.config_path)
