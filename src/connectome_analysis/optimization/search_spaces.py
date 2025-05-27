from typing import Dict, Any

def get_baseline_search_space() -> Dict[str, Dict[str, Any]]:
    """
    Defines the hyperparameter search space for baseline models (e.g., SVM, Logistic Regression).
    """
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        # Example for SVM:
        "model_params.C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "model_params.kernel": {"type": "categorical", "choices": ["linear", "rbf"]},
        # Example for Logistic Regression:
        "model_params.solver": {"type": "categorical", "choices": ["liblinear", "lbfgs"]},
        "model_params.penalty": {"type": "categorical", "choices": ["l1", "l2"]},
    }

def get_gnn_search_space() -> Dict[str, Dict[str, Any]]:
    """
    Defines the hyperparameter search space for GNN models (e.g., GCN, GAT).
    """
    return {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        "model_params.hidden_channels": {"type": "categorical", "choices": [32, 64, 128]},
        "model_params.num_layers": {"type": "int", "low": 2, "high": 4},
        "model_params.dropout": {"type": "float", "low": 0.2, "high": 0.5},
        # GAT specific:
        "model_params.heads": {"type": "categorical", "choices": [1, 2, 4]},
    }

def get_transformer_search_space() -> Dict[str, Dict[str, Any]]:
    """
    Defines the hyperparameter search space for Transformer models.
    """
    return {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32]},
        "model_params.d_model": {"type": "categorical", "choices": [64, 128, 256]},
        "model_params.nhead": {"type": "categorical", "choices": [4, 8]},
        "model_params.num_encoder_layers": {"type": "int", "low": 2, "high": 4},
        "model_params.dropout": {"type": "float", "low": 0.1, "high": 0.3},
    }

def get_search_space(model_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns the appropriate hyperparameter search space for a given model type.
    """
    if model_type == "baseline":
        return get_baseline_search_space()
    elif model_type == "gnn":
        return get_gnn_search_space()
    elif model_type == "transformer":
        return get_transformer_search_space()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
