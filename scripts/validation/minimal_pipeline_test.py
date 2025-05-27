#!/usr/bin/env python3
"""
Minimal working example demonstrating the complete CGN pipeline
Tests the integration with a small subset of ABIDE data
"""

import numpy as np
import torch
import os
import sys
import time
from typing import Dict, Any, List, Tuple, Optional, cast, Union, overload, Literal
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import necessary components
from src.connectome_analysis.data.data_modules import ConnectomeDataModule # Import ConnectomeDataModule
from src.connectome_analysis.evaluation.cross_validation import CrossValidator # Import CrossValidator
from src.connectome_analysis.training.baseline_trainer import BaselineLightningModule # Import BaselineTrainer
from src.connectome_analysis.training.gnn_trainer import GNNLightningModule # Import GNNTrainer
from src.connectome_analysis.models.baseline_models import SVMClassifier, RandomForest, LogisticRegressionClassifier # Import actual baseline models
from src.connectome_analysis.models.gnn_models import BrainGCN, BrainGAT # Import actual GNN models
from src.connectome_analysis.evaluation.model_comparison import generate_full_report
from src.connectome_analysis.evaluation.metrics import calculate_classification_metrics
from src.connectome_analysis.evaluation.reporting import generate_results_report
from sklearn.metrics import roc_curve, auc
import nilearn.datasets # Import nilearn.datasets for fetching atlas
from scripts.integration.prepare_abide_data import load_and_process_abide, create_baseline_features, create_graph_data

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Target scores, can either be probability estimates of the positive class,
                              confidence values, or non-thresholded decision values.
        title (str): Title of the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

# Model builder functions using the factory pattern
def build_baseline_model_for_cv(model_name: str, model_params: Dict[str, Any]):
    """Helper function to build a BaselineLightningModule instance for CrossValidator."""
    # Instantiate the actual baseline model
    if model_name == 'SVMClassifier':
        model = SVMClassifier(**model_params)
    elif model_name == 'RandomForest':
        model = RandomForest(**model_params)
    elif model_name == 'LogisticRegressionClassifier':
        model = LogisticRegressionClassifier(**model_params)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    
    # BaselineLightningModule expects the instantiated model, num_classes, and learning_rate
    # Assuming binary classification (2 classes) and a default learning rate
    return BaselineLightningModule(model, num_classes=2, learning_rate=0.001)

def build_gnn_model_for_cv(model_name: str, model_config: Dict[str, Any], trainer_params: Dict[str, Any]):
    """Helper function to build a GNNLightningModule instance for CrossValidator."""
    # Instantiate the actual GNN model
    if model_name == 'BrainGCN':
        model = BrainGCN(**model_config)
    elif model_name == 'BrainGAT':
        model = BrainGAT(**model_config)
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")

    # GNNLightningModule expects the instantiated model, num_classes, and learning_rate
    # Assuming binary classification (2 classes) and learning rate from trainer_params
    learning_rate = trainer_params.get('optimizer', {}).get('lr', 0.001)
    return GNNLightningModule(model, num_classes=2, learning_rate=learning_rate)


def minimal_pipeline_test():
    """
    Complete end-to-end test with ~50 ABIDE subjects:
    1. Load small ABIDE subset using existing data loaders
    2. Create connectivity matrices using existing connectome builder
    3. Train one baseline model (SVM) and one GNN model (GCN)
    4. Evaluate using proper cross-validation
    5. Generate statistical comparison
    6. Create simple visualization
    7. Report results and timing
    """
    print("Starting minimal pipeline test...")
    start_time = time.time()

    # Configuration for the minimal test
    test_config = {
        'data': {
            'dataset_name': 'abide',
            'data_dir': os.path.join(os.getcwd(), 'data', 'raw'), # Use raw data directory for loading
            'processed_data_dir': os.path.join(os.getcwd(), 'data', 'processed', 'abide_test_data'), # Directory for processed data
            'num_subjects_per_class': 5 # Small subset (this is handled by data_module internally if needed)
        },
        'cross_validation': {
            'cv_folds': 3, # Reduced folds for faster test
            'stratify_by_site': True, # Test site stratification
            'random_state': 42,
            'batch_size': 16, # Reduced batch size for dummy data
            'max_epochs': 5, # Very short training for speed
            'accelerator': 'cpu', # Force CPU for minimal test
            'devices': 1, # Force 1 device for CPU
            'accumulate_grad_batches': 1 # No gradient accumulation for small test
        },
        'models': {
            'svm': {
                'model_type': 'baseline',
                'model_name': 'SVMClassifier', # Use the actual class name from baseline_models.py
                'model_params': {'C': 1.0, 'kernel': 'linear'},
                'trainer_params': {} # BaselineTrainer doesn't use these directly, but CrossValidator passes them
            },
            'gcn': {
                'model_type': 'gnn',
                'model_name': 'BrainGCN', # Use the actual class name from gnn_models.py
                'model_params': {
                    'in_channels': 16, # Matches dummy node feature dimension (regions x 16)
                    'hidden_channels': 32,
                    'out_channels': 2, # Output channels for binary classification (2 classes)
                },
                'trainer_params': { # These are hparams for ConnectomeTrainer and trainer_kwargs for pl.Trainer
                    'optimizer': {'type': 'Adam', 'lr': 0.001},
                    'max_epochs': 5, # Very short training for speed
                    'accelerator': 'cpu',
                    'devices': 1,
                    'accumulate_grad_batches': 1
                }
            }
        },
        'evaluation': {
            'alpha': 0.05,
            'correction_method': 'fdr'
        },
        'output_dir': 'results/minimal_test_run' # Directory to save results
    }

    # 1. Load ABIDE data and create connectome features
    print("\nLoading and processing ABIDE data to create connectome features...")
    data_dir = test_config['data']['data_dir']
    processed_data_dir = test_config['data']['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)

    # Fetch MSDL atlas (required for connectome building)
    print("Fetching MSDL atlas...")
    try:
        msdl_atlas = nilearn.datasets.fetch_atlas_msdl()
        atlas_path = msdl_atlas['maps']
        atlas_coords = np.array(msdl_atlas['region_coords'])
        print(f"MSDL atlas fetched. Atlas path: {atlas_path}")
    except Exception as e:
        print(f"Error fetching MSDL atlas: {e}. Please ensure nilearn is installed and you have an internet connection.")
        print("Exiting as atlas is required for connectome building.")
        return # Exit the function if atlas fetching fails

    connectomes, labels, sites, subject_ids = load_and_process_abide(data_dir, atlas_path, atlas_coords)
    
    if not connectomes:
        print("No connectomes processed successfully. Exiting minimal pipeline test.")
        return

    labels_np = np.array(labels)
    sites_np = np.array(sites)

    print("Creating baseline features...")
    baseline_features = create_baseline_features(connectomes)
    print(f"Baseline features shape: {baseline_features.shape}")

    print("Creating graph data...")
    graph_data_list = create_graph_data(connectomes, labels, atlas_coords=atlas_coords)
    print(f"Number of graph data objects: {len(graph_data_list)}")


    # 3. & 4. Train and Evaluate models using Cross-Validation
    all_model_results: Dict[str, List[Dict[str, Any]]] = {}

    # Evaluate Baseline Model (SVM)
    print("\nRunning CV for Baseline Model (SVM)...")
    svm_model_spec = test_config['models']['svm']
    svm_cv_config = test_config['cross_validation'].copy()
    svm_cv_config.update(svm_model_spec) # Combine configs
    
    # Add model_builder and data_module_params to the config for CrossValidator
    svm_cv_config['model_builder'] = lambda: build_baseline_model_for_cv(svm_model_spec['model_name'], svm_model_spec['model_params'])
    svm_cv_config['data_module_params'] = {
        'features': baseline_features,
        'labels': labels_np,
        'site_labels': sites_np,
        'batch_size': svm_cv_config['batch_size'],
        'random_state': svm_cv_config['random_state']
    }
    
    svm_cv_runner = CrossValidator(config=svm_cv_config)
    svm_results = svm_cv_runner.run_cross_validation(features=baseline_features, labels=labels_np, site_labels=sites_np)
    all_model_results.update({svm_model_spec['model_name']: svm_results}) # Use model_name as key

    # Evaluate GNN Model (GCN)
    print("\nRunning CV for GNN Model (GCN)...")
    gnn_model_spec = test_config['models']['gcn']
    gnn_cv_config = test_config['cross_validation'].copy()
    gnn_cv_config.update(gnn_model_spec) # Combine configs
    
    # Add model_builder and data_module_params to the config for CrossValidator
    gnn_cv_config['model_builder'] = lambda: build_gnn_model_for_cv(
        gnn_model_spec['model_name'],
        gnn_model_spec['model_params'],
        gnn_model_spec['trainer_params']
    )
    gnn_cv_config['data_module_params'] = {
        'features': graph_data_list, # Pass the list of PyG Data objects
        'labels': labels_np,
        'site_labels': sites_np,
        'batch_size': gnn_cv_config['batch_size'],
        'random_state': gnn_cv_config['random_state']
    }

    gnn_cv_runner = CrossValidator(config=gnn_cv_config)
    gnn_results = gnn_cv_runner.run_cross_validation(features=graph_data_list, labels=labels_np, site_labels=sites_np)
    all_model_results.update({gnn_model_spec['model_name']: gnn_results}) # Use model_name as key


    # 5. Generate statistical comparison
    print("\nGenerating Statistical Comparison...")
    auc_results_for_stat_test: Dict[str, List[float]] = {}
    for model_name, fold_results_list in all_model_results.items():
         auc_scores = []
         for fold_result_dict in fold_results_list:
              if fold_result_dict and isinstance(fold_result_dict, dict):
                   fold_name = list(fold_result_dict.keys())[0]
                   fold_metrics = fold_result_dict[fold_name]
                   if fold_metrics is not None and 'auc' in fold_metrics and not np.isnan(fold_metrics['auc']):
                        auc_scores.append(fold_metrics['auc'])
         if auc_scores:
              auc_results_for_stat_test[model_name] = auc_scores


    stat_comparison_results = None
    if len(auc_results_for_stat_test) >= 2:
        try:
            model_results_for_comparison = {}
            for model_name, auc_scores in auc_results_for_stat_test.items():
                model_results_for_comparison[model_name] = [{'auc': score} for score in auc_scores]

            # Assuming generate_full_report returns a dict with 'model_comparison_table'
            full_report = generate_full_report(
                model_results_for_comparison,
                primary_metric='auc',
                baseline_model_name=None, # No baseline for this test
                n_permutations=1000,
                random_state=42
            )
            stat_comparison_results = full_report.get('model_comparison_table')
            print("\nStatistical Comparison Results:")
            print(stat_comparison_results)

        except Exception as e:
            print(f"Error during statistical comparison: {e}")
            stat_comparison_results = {'error': f"Error during statistical comparison: {e}"}
    else:
        print("\nNot enough models with valid AUC scores for statistical comparison.")
        stat_comparison_results = {'message': "Not enough models for statistical comparison."}


    # 6. Create simple visualization (ROC curves)
    print("\nGenerating ROC Curve Visualization...")
    output_dir = test_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    roc_plot_path = os.path.join(output_dir, "minimal_pipeline_roc_curves.png")

    try:
        y_true_example = None
        y_proba_example = None
        
        first_model_name = next(iter(all_model_results))
        if all_model_results[first_model_name]:
            first_fold_results = all_model_results[first_model_name][0]
            if first_fold_results and isinstance(first_fold_results, dict):
                fold_name = list(first_fold_results.keys())[0]
                fold_result_metrics = first_fold_results[fold_name]
                y_true_example = fold_result_metrics.get('y_true')
                y_proba_example = fold_result_metrics.get('y_prob')

        if y_true_example is not None and y_proba_example is not None:
            plot_roc_curve(np.array(y_true_example), np.array(y_proba_example), title=f"ROC Curve for {first_model_name} (First Fold)")
            plt.savefig(roc_plot_path)
            plt.close()
        else:
            print("No y_true and y_proba found for ROC plotting in the first fold of the first model.")

    except Exception as e:
        print(f"Error generating ROC curve plot: {e}")


    # 7. Report results and timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nMinimal pipeline test finished in {total_time:.2f} seconds.")

    print("\n--- Minimal Pipeline Test Summary ---")
    print(f"Configuration: {test_config['data']['dataset_name']} dataset, {test_config['cross_validation']['cv_folds']} folds")
    print(f"Output directory: {output_dir}")

    print("\nCross-Validation Results:")
    for model_name, fold_results_list in all_model_results.items():
         if fold_results_list and "error" not in fold_results_list[0]:
              metrics_list = []
              for fold_result_dict in fold_results_list:
                   if fold_result_dict and isinstance(fold_result_dict, dict):
                        fold_name = list(fold_result_dict.keys())[0]
                        fold_metrics = fold_result_dict[fold_name]
                        metrics_list.append(fold_metrics)

              if metrics_list:
                   df_results = pd.DataFrame(metrics_list)
                   print(f"\n{model_name} Metrics (Mean ± Std. Dev.):")
                   for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
                        if metric in df_results.columns:
                             mean = df_results[metric].mean()
                             std = df_results[metric].std()
                             print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
              else:
                   print(f"\nNo valid metrics found for {model_name}.")
         elif fold_results_list and "error" in fold_results_list[0]:
              print(f"\n{model_name} failed during CV: {fold_results_list[0]['error']}")
         else:
              print(f"\nNo results recorded for {model_name}.")


    print("\nStatistical Comparison Results:")
    if isinstance(stat_comparison_results, pd.DataFrame) and not stat_comparison_results.empty:
        print(stat_comparison_results)
    elif isinstance(stat_comparison_results, dict) and stat_comparison_results.get('error'):
         print(f"Statistical comparison failed: {stat_comparison_results['error']}")
    else:
         print("Statistical comparison was not performed (requires at least two models with valid AUC scores).")


    print(f"\nROC curve plot saved to: {roc_plot_path}")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    try:
        minimal_pipeline_test()
    except Exception as e:
        print(f"An error occurred during minimal pipeline test: {e}")
