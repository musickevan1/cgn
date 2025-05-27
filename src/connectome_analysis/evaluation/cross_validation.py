import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, cast
from src.connectome_analysis.data.dataset_splits import get_data_split_indices
from src.connectome_analysis.data.data_modules import ConnectomeDataModule
from src.connectome_analysis.training.experiment_manager import ExperimentManager
from src.connectome_analysis.evaluation.metrics import calculate_classification_metrics
import torch
import pytorch_lightning as pl # Added import
from torch_geometric.data import Data # Import Data object
import traceback # Added import for detailed traceback

class CrossValidator:
    """
    Performs cross-validation using PyTorch Lightning models and data modules.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CrossValidator.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - 'cv_folds' (int): Number of cross-validation folds.
                - 'split_type' (str): Type of split ('stratified_kfold' or 'leave_one_site_out').
                - 'stratify_by_site' (bool): Whether to stratify by site (for ABIDE).
                - 'random_state' (int, optional): Random state for reproducibility.
                - 'model_type' (str): Type of model ('baseline', 'gnn', 'transformer').
                - 'model_name' (str): Name of the model for logging.
                - 'trainer_params' (Dict): Parameters for the PyTorch Lightning Trainer.
                - 'data_module_params' (Dict): Parameters for the ConnectomeDataModule.
                - 'model_builder' (callable): Function that builds and returns a model instance.
        """
        self.config = config
        self.n_splits = config.get('cv_folds', 5)
        self.split_type = config.get('split_type', 'stratified_kfold')
        self.stratify_by_site = config.get('stratify_by_site', False)
        self.random_state = config.get('random_state', None)
        self.model_type = config.get('model_type', 'baseline')
        self.model_name = config.get('model_name', 'default_model')
        self.trainer_params = config.get('trainer_params', {})
        self.data_module_params = config.get('data_module_params', {})
        
        _model_builder = config.get('model_builder')
        if _model_builder is None:
            raise ValueError("model_builder function must be provided in the config.")
        self.model_builder: Callable = cast(Callable, _model_builder)
        assert callable(self.model_builder), "model_builder must be a callable function."

    def run_cross_validation(
        self,
        features: Union[np.ndarray, List[Data]], # Allow List[Data]
        labels: np.ndarray,
        site_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Runs the cross-validation process.

        Args:
            features (Union[np.ndarray, List[Data]]): The input features for the models.
            labels (np.ndarray): The labels corresponding to the features.
            site_labels (np.ndarray, optional): Site labels for site-aware splitting.

        Returns:
            Dict[str, Any]: A dictionary containing aggregated results from all folds.
        """
        all_fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        # Determine the labels to use for splitting (main labels or site labels)
        if self.split_type == "leave_one_site_out" and site_labels is None:
            raise ValueError("site_labels must be provided for 'leave_one_site_out' split type.")
        
        # Get the split indices
        splits = get_data_split_indices(
            labels=labels,
            site_labels=site_labels if self.split_type == "leave_one_site_out" else None,
            split_type=self.split_type,
            n_splits=self.n_splits,
            shuffle=True, # Always shuffle for stratified k-fold
            random_state=self.random_state
        )

        if not splits:
            raise ValueError("No valid splits generated. Check your data and split parameters.")

        print(f"Running {len(splits)} cross-validation folds for {self.model_name} ({self.model_type})...")

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")

            # Initialize DataModule for the current fold
            # Ensure data_module_params has fold_idx, as it's loop-specific
            current_fold_data_module_params = self.data_module_params.copy()
            current_fold_data_module_params['fold_idx'] = fold_idx
            # Other parameters like split_type, n_splits, random_state should already be in self.data_module_params

            # site_labels is already part of current_fold_data_module_params if needed
            fold_data_module = ConnectomeDataModule(
                features=features,
                labels=labels,
                # Pass other necessary params explicitly if not in data_module_params or if they override
                **current_fold_data_module_params
            )
            fold_data_module.setup() # Call setup to create train/val/test datasets

            # Build model for the current fold
            model_instance = self.model_builder() # Model builder should handle its own parameters

            # Initialize ExperimentManager for the current fold
            experiment_config = {
                "model_type": self.model_type,
                "model_name": f"{self.model_name}_fold_{fold_idx}",
                "num_classes": self.config.get("num_classes", 2),
                "learning_rate": self.config.get("learning_rate", 1e-3),
                "epochs": self.config.get("epochs", 10),
                "batch_size": self.config.get("batch_size", 32),
                "gpus": self.config.get("gpus", 0),
                "log_dir": self.config.get("log_dir", "lightning_logs"),
                "checkpoint_dir": self.config.get("checkpoint_dir", "checkpoints"),
                **self.trainer_params # Override with specific trainer params
            }
            experiment_manager = ExperimentManager(experiment_config)

            # Run training and testing for the current fold
            try:
                # experiment_manager.run_experiment now returns (metrics_dict_list, lightning_module_instance)
                # experiment_manager.run_experiment now returns (metrics_dict_list, lightning_module_instance)
                test_run_output, executed_lightning_module = experiment_manager.run_experiment(model_instance, fold_data_module)
                
                fold_metrics: Dict[str, Any] = {} # Explicitly type hint
                # trainer.test() returns a list of dicts, one for each test dataloader. We expect one.
                if test_run_output and isinstance(test_run_output, list) and len(test_run_output) > 0 and isinstance(test_run_output[0], dict):
                    # Use the metrics logged by trainer.test()
                    fold_metrics = {k: v for k, v in test_run_output[0].items() if k.startswith('test_')}
                    if not fold_metrics: # If it's an empty dict or only non-test_ metrics
                        print(f"Warning: No metrics starting with 'test_' found in test_run_output for fold {fold_idx + 1}. Full output: {test_run_output[0]}")
                        # Keep fold_metrics potentially empty, or mark as error if appropriate
                        # fold_metrics = {"error": "No test_ metrics found"} # Option
                else:
                    print(f"Warning: Invalid or empty test results returned by ExperimentManager for fold {fold_idx + 1}. Output: {test_run_output}")
                    fold_metrics = {"error": "Invalid or empty test results from ExperimentManager"}

                y_true_fold = np.array([])
                y_pred_fold = np.array([])
                y_prob_fold = np.array([]) # For AUROC calculation, typically prob of positive class

                # Retrieve raw predictions from the executed_lightning_module
                if hasattr(executed_lightning_module, 'all_test_targets') and executed_lightning_module.all_test_targets:
                    y_true_fold = torch.cat(executed_lightning_module.all_test_targets).cpu().numpy()
                else:
                    print(f"Warning: No 'all_test_targets' collected or list is empty in executed_lightning_module for fold {fold_idx + 1}.")
                    # Ensure fold_metrics reflects this error if not already set
                    if "error" not in fold_metrics: fold_metrics["error"] = "No test targets collected"


                if hasattr(executed_lightning_module, 'all_test_preds') and executed_lightning_module.all_test_preds:
                    y_pred_fold = torch.cat(executed_lightning_module.all_test_preds).cpu().numpy()
                else:
                    print(f"Warning: No 'all_test_preds' collected or list is empty in executed_lightning_module for fold {fold_idx + 1}.")
                    if "error" not in fold_metrics: fold_metrics["error"] = "No test predictions collected"

                if hasattr(executed_lightning_module, 'all_test_logits') and executed_lightning_module.all_test_logits:
                    all_logits_tensor = torch.cat(executed_lightning_module.all_test_logits).cpu()
                    if executed_lightning_module.num_classes == 2:
                        # For binary, AUROC often uses probability of the positive class.
                        # If all_test_logits are true logits, apply sigmoid. If they are already probabilities (as in Baseline), use as is.
                        # BaseLightningModule's test_step stores the direct output of _common_step's `output` as logits.
                        # For Baseline, this `output` is probabilities.
                        if self.model_type == "baseline" and hasattr(model_instance, "predict_proba"): # It's probabilities
                             y_prob_fold = all_logits_tensor[:, 1].numpy() if all_logits_tensor.ndim > 1 and all_logits_tensor.shape[1] > 1 else all_logits_tensor.numpy()
                        else: # Assume true logits for GNN/Transformer
                             y_prob_fold = torch.softmax(all_logits_tensor, dim=1)[:, 1].numpy() if all_logits_tensor.ndim > 1 and all_logits_tensor.shape[1] > 1 else torch.sigmoid(all_logits_tensor).numpy()
                    else: # Multiclass
                        y_prob_fold = torch.softmax(all_logits_tensor, dim=1).numpy() # Store all class probabilities
                else:
                    print(f"Warning: No 'all_test_logits' found or empty in executed_lightning_module for fold {fold_idx + 1}.")
                
                # Important: Clear the collected predictions from the module instance for the next fold if this instance is reused.
                # However, ExperimentManager creates a new lightning_module for each call to run_experiment,
                # and CrossValidator creates a new ExperimentManager per fold. So, this might not be strictly necessary here,
                # but good practice if the module were reused.
                # executed_lightning_module.all_test_preds.clear()
                # executed_lightning_module.all_test_targets.clear()
                # executed_lightning_module.all_test_logits.clear()

                all_fold_results.append(fold_metrics)
                if y_true_fold.size > 0: all_y_true.extend(y_true_fold.tolist())
                if y_pred_fold.size > 0: all_y_pred.extend(y_pred_fold.tolist())
                if y_prob_fold.size > 0: all_y_prob.extend(y_prob_fold.tolist()) # y_prob_fold might be 2D for multiclass

            except Exception as e:
                print(f"Error during training/testing for fold {fold_idx + 1}: {e}")
                print(traceback.format_exc()) # Print full traceback
                all_fold_results.append({"error": str(e)})

        # Aggregate results
        aggregated_metrics = self._aggregate_results(all_fold_results)
        
        # Calculate overall metrics if y_true, y_pred, y_prob were collected
        if all_y_true and all_y_pred:
            overall_metrics = calculate_classification_metrics(
                np.array(all_y_true),
                np.array(all_y_pred),
                np.array(all_y_prob) if all_y_prob else None
            )
            aggregated_metrics['overall_metrics'] = overall_metrics

        return aggregated_metrics

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates metrics from all folds."""
        aggregated = {}
        metric_keys = set()
        for res in results:
            for k in res.keys():
                if not k.startswith('error'): # Exclude error messages from aggregation
                    metric_keys.add(k)
        
        for key in metric_keys:
            values = [res[key] for res in results if key in res and not isinstance(res[key], str)] # Filter out errors
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
        
        return aggregated
