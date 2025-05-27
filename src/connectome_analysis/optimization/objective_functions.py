from typing import Dict, Any, Tuple, Callable, Optional
import optuna

def multi_objective_wrapper(
    objective_func: Callable,
    weights: Dict[str, float]
) -> Callable:
    """
    Wraps a single-objective Optuna objective function to create a multi-objective one.
    The wrapper combines multiple metrics into a single scalar value using weighted sum.

    Args:
        objective_func (callable): The original single-objective function that returns
                                   a dictionary of metrics (e.g., {'accuracy': 0.8, 'f1': 0.7}).
        weights (Dict[str, float]): A dictionary mapping metric names to their weights.
                                    Higher weight means more importance.
                                    Metrics to be maximized should have positive weights.
                                    Metrics to be minimized should have negative weights.

    Returns:
        callable: A new objective function suitable for Optuna, returning a single float.
    """
    def wrapped_objective(trial: optuna.Trial) -> float:
        metrics = objective_func(trial) # Call the original objective function
        
        if not isinstance(metrics, dict):
            raise TypeError("Original objective function must return a dictionary of metrics.")

        combined_score = 0.0
        for metric_name, weight in weights.items():
            if metric_name not in metrics:
                raise ValueError(f"Metric '{metric_name}' not found in objective function's return.")
            combined_score += metrics[metric_name] * weight
        
        return combined_score
    return wrapped_objective

def get_multi_objective_weights(
    primary_metric: str = "val_auroc",
    secondary_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Generates a dictionary of weights for multi-objective optimization.

    Args:
        primary_metric (str): The main metric to optimize (e.g., 'val_auroc', 'val_accuracy').
                              This metric will have a weight of 1.0.
        secondary_metrics (Dict[str, float], optional): A dictionary of secondary metrics
                                                        and their relative weights.
                                                        Example: {'val_f1': 0.5, 'val_loss': -0.1}.
                                                        Use negative weights for metrics to be minimized.

    Returns:
        Dict[str, float]: A dictionary mapping metric names to their weights.
    """
    weights = {primary_metric: 1.0}
    if secondary_metrics:
        weights.update(secondary_metrics)
    return weights

# Example usage in optuna_integration.py (conceptual)
# from src.connectome_analysis.optimization.objective_functions import multi_objective_wrapper, get_multi_objective_weights
#
# class OptunaOptimizer:
#     def objective(self, trial: optuna.Trial) -> float:
#         # ... (model building, training, validation)
#         # Instead of returning a single metric, return a dict of metrics
#         # eval_results = trainer.validate(lightning_module, data_module.val_dataloader())
#         # val_auroc = eval_results[0].get('val_auroc', 0.0)
#         # val_loss = eval_results[0].get('val_loss', 0.0)
#         # return {'val_auroc': val_auroc, 'val_loss': val_loss}
#
#     def optimize(self) -> optuna.Study:
#         # ...
#         # Define weights for multi-objective optimization
#         weights = get_multi_objective_weights(
#             primary_metric="val_auroc",
#             secondary_metrics={"val_loss": -0.1} # Minimize loss
#         )
#         wrapped_objective = multi_objective_wrapper(self.objective, weights)
#         study = optuna.create_study(
#             study_name=self.study_name,
#             directions=['maximize'], # Or 'minimize' if combined_score is designed for minimization
#             pruner=optuna.pruners.MedianPruner(...)
#         )
#         study.optimize(wrapped_objective, n_trials=self.n_trials)
#         return study
