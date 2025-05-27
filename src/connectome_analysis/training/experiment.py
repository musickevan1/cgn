# src/connectome_analysis/training/experiment.py

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import pickle # Import pickle
from typing import Dict, Any, List, Optional

# Import components from trainer.py
from src.connectome_analysis.training.trainer import ConnectomeDataset, CrossValidator, ModelOptimizer

# Import model factory functions
from src.connectome_analysis.models import (
    create_baseline_classifier,
    create_gnn_model
)

# Import evaluation metrics and reporting functions
from src.connectome_analysis.evaluation.metrics import calculate_classification_metrics # Use the new name
from src.connectome_analysis.evaluation import statistical_tests # Import statistical_tests module
from src.connectome_analysis.evaluation.metrics import plot_roc_curve # Use the new name


def run_experiment(data_path: str, output_dir: str, config_path: str):
    """
    Runs a complete connectome analysis experiment based on a configuration file.

    Args:
        data_path: Path to the directory containing processed data (features, labels, sites).
        output_dir: Directory to save experiment results.
        config_path: Path to the experiment configuration YAML file.
    """
    print(f"Starting experiment with config: {config_path}")

    # 1. Load experiment configuration
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            experiment_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading experiment config from {config_path}: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load data
    print(f"Loading processed data from {data_path}...")
    # The experiment config should specify the model type to load appropriate data.
    model_type = experiment_config.get('model_type', 'baseline') # Default to baseline
    try:
        # Assuming data loading scripts save data in a specific format (e.g., pkl files)
        # The format depends on whether it's baseline (flattened) or GNN (graph) data.

        if model_type == 'baseline':
            data_file = "baseline_features.pkl"
        elif model_type == 'gnn' or model_type == 'transformer': # Transformer models also use graph data
            data_file = "graph_data_list.pkl" # Assuming graph data is saved as a list of PyG Data objects
        else:
            print(f"Error: Unknown model_type '{model_type}' specified in config.")
            return

        with open(os.path.join(data_path, data_file), "rb") as f:
            data = pickle.load(f)
        with open(os.path.join(data_path, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)
        with open(os.path.join(data_path, "sites.pkl"), "rb") as f:
            sites = pickle.load(f)

        # Create ConnectomeDataset
        dataset = ConnectomeDataset(data, labels, sites, model_type=model_type)
        print(f"Loaded dataset with {len(dataset)} subjects for model type '{model_type}'.")

    except FileNotFoundError:
        print(f"Error: Required data files not found in {data_path} for model type '{model_type}'. Please ensure data preprocessing is complete.")
        return
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return

    # 3. Run experiments for each model specified in the config
    all_model_results: Dict[str, List[Dict[str, Any]]] = {}
    models_config = experiment_config.get('models', {})

    if not models_config:
        print("No models specified in the configuration. Exiting.")
        return

    for model_name, model_config in models_config.items():
        print(f"\n--- Running experiment for model: {model_name} ---")

        # Create a model builder function for this specific model
        def current_model_builder(params_config):
            if model_type == 'baseline':
                # For baseline models, model_name directly maps to the classifier type (e.g., 'SVMClassifier')
                return create_baseline_classifier(model_name, params_config)
            elif model_type == 'gnn':
                # For GNN models, model_name maps to the GNN architecture (e.g., 'BrainGCN')
                # The params_config should contain 'in_channels', 'hidden_channels', 'out_channels', 'num_classes', etc.
                return create_gnn_model(model_name, params_config)
            elif model_type == 'transformer':
                # For Transformer models, model_name maps to the transformer architecture (e.g., 'BrainGraphTransformer')
                # The params_config should contain 'in_channels', 'd_model', 'nhead', etc.
                from src.connectome_analysis.models import create_transformer_model # Import here to avoid circular dependency
                return create_transformer_model(model_name, params_config)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

        # Prepare configuration for the CrossValidator
        cv_config = experiment_config.copy() # Start with overall experiment config
        # model_type is guaranteed to be bound here due to the check after loading config
        cv_config['model_type'] = model_type # Ensure model type is passed # type: ignore[possibly-unbound]
        cv_config['model_name'] = model_name # Pass the specific model name
        cv_config['model_params'] = model_config.get('model_params', {}) # Pass model-specific parameters
        cv_config['trainer_params'] = model_config.get('trainer_params', {}) # Pass trainer parameters

        # Optional: Hyperparameter optimization (Placeholder)
        # If 'optimization' is in model_config, run Optuna before CV
        # if 'optimization' in model_config:
        #     print(f"Running hyperparameter optimization for {model_name}...")
        #     optuna_config = experiment_config.get('optimization', {}) # Use overall optimization config
        #     optuna_config.update(model_config.get('optimization', {})) # Override with model-specific optimization config
        #     optimizer = ModelOptimizer(current_model_builder, dataset, optuna_config)
        #     study = optimizer.optimize()
        #     best_params = study.best_trial.params
        #     print(f"Best hyperparameters for {model_name}: {best_params}")
        #     # Update model_params for CV with best_params
        #     cv_config['model_params'].update(best_params)


        # Initialize and run CrossValidator
        try:
            cross_validator = CrossValidator(
                model_builder=current_model_builder,
                dataset=dataset,
                config=cv_config
            )
            model_results = cross_validator.run_cross_validation()
            all_model_results.update(model_results) # Add results to the overall dictionary

        except Exception as e:
            print(f"Error running cross-validation for {model_name}: {e}")
            all_model_results[model_name] = [{'error': f"Error running CV: {e}"}] # Log error


    # 4. Generate comprehensive results report (Placeholder for now)
    print("\n--- Generating Results Report (Placeholder) ---")
    # The actual reporting will be implemented in a dedicated reporting module.
    # For now, we'll just print a message.
    print("Reporting functionality will be implemented in a separate module.")


    print("\n--- Experiment Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run connectome analysis experiments.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing processed data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save experiment results.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the experiment configuration YAML file.")

    args = parser.parse_args()

    run_experiment(args.data_path, args.output_dir, args.config_path)
