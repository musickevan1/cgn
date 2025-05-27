# src/connectome_analysis/models/baseline.py

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Union, Tuple


def extract_basic_matrix_features(matrix: np.ndarray) -> np.ndarray:
    """
    Extracts basic features from a connectivity matrix (e.g., degree).

    Args:
        matrix: A square numpy array representing the connectivity matrix.

    Returns:
        A 1D numpy array containing basic features (e.g., sum of connections per node).
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Calculate degree (sum of connections for each node)
    degree = np.sum(matrix, axis=1)

    # You could add other basic features here, e.g., variance of connections per node
    # variance = np.var(matrix, axis=1)
    # return np.concatenate([degree, variance])

    return degree # Returning just degree for simplicity


# Dictionary to map feature extraction method names to functions
FEATURE_EXTRACTION_FUNCTIONS = {
    'basic_features': extract_basic_matrix_features,
}

# Dictionary to map model names to their classes for safer instantiation
MODEL_CLASSES = {
    'SVC': SVC,
    'RandomForestClassifier': RFClassifier,
    'LogisticRegression': LogisticRegression
}

class BaselineClassifier:
    """Base class for classical ML classifiers on connectome data"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.pipeline = None

    def build_pipeline(self):
        """Build the scikit-learn pipeline with preprocessing and model"""
        steps: List = []

        # Optional Feature Extraction step
        if 'feature_extraction' in self.config:
            fe_config = self.config['feature_extraction']
            method = fe_config.get('method')
            if method in FEATURE_EXTRACTION_FUNCTIONS:
                # Using FunctionTransformer to integrate the extraction function into the pipeline
                steps.append((f'{method}_features', FunctionTransformer(func=FEATURE_EXTRACTION_FUNCTIONS[method], validate=False)))
            else:
                raise ValueError(f"Unknown feature extraction method: {method}")

        # Feature Scaling
        steps.append(('scaler', StandardScaler()))

        # Feature Selection (optional, based on config)
        if 'feature_selection' in self.config:
            fs_config = self.config['feature_selection']
            if fs_config['method'] == 'variance_threshold':
                steps.append(('variance_threshold', VarianceThreshold(threshold=fs_config.get('threshold', 0.0))))
            elif fs_config['method'] == 'select_k_best':
                steps.append(('select_k_best', SelectKBest(score_func=f_classif, k=fs_config.get('k', 10))))
            elif fs_config['method'] == 'mutual_info_classif':
                 steps.append(('mutual_info_classif', SelectKBest(score_func=mutual_info_classif, k=fs_config.get('k', 10))))

        # Model
        steps.append(('model', self.model))

        self.pipeline = Pipeline(steps)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier"""
        self.build_pipeline() # Ensure pipeline is built
        if self.pipeline is None:
             raise RuntimeError("Pipeline failed to build.")
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data"""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline not built or trained.")
        result = self.pipeline.predict(X)
        if isinstance(result, tuple):
            # If the result is a tuple, assume predictions are in the first element
            return np.array(result[0])
        return np.array(result)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data. Returns probabilities for the positive class."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline not built or trained.")

        if hasattr(self.pipeline, 'predict_proba'):
            raw_predictions = self.pipeline.predict_proba(X)

            if isinstance(raw_predictions, tuple):
                # For VotingClassifier with 'soft' voting, it returns a tuple of probabilities
                # We assume the first element contains the aggregated probabilities
                probabilities = raw_predictions[0]
            else:
                probabilities = raw_predictions

            # Ensure probabilities is a 2D array (n_samples, n_classes)
            if probabilities.ndim == 1:
                # If it's 1D, it might be probabilities for a single class, or needs reshaping
                # This case is less common for predict_proba, but handled for robustness
                return probabilities
            elif probabilities.shape[1] == 2:
                # For binary classification, return probabilities of the positive class
                return probabilities[:, 1]
            else:
                # For multi-class, return the full probability array
                return probabilities
        else:
            # Fallback for models without predict_proba (e.g., SVC without probability=True)
            # This will return 0 or 1 based on prediction, not actual probabilities
            predictions = self.pipeline.predict(X)
            # Create a 1D array of 0s and 1s for binary classification
            return np.array(predictions == 1).astype(float)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the classifier"""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1] # Probability of the positive class # type: ignore

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'auc': roc_auc_score(y, y_prob),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_prob),
            'f1': f1_score(y, y_pred)
        }
        return metrics

class SVMClassifier(BaselineClassifier):
    """Support Vector Machine for connectome classification"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = SVC(probability=True, **self.config.get('params', {}))
        self.build_pipeline()

class RandomForestClassifier(BaselineClassifier):
    """Random Forest for connectome classification"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = RFClassifier(**self.config.get('params', {}))
        self.build_pipeline()

class LogisticRegressionClassifier(BaselineClassifier):
    """Logistic Regression for connectome classification"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = LogisticRegression(**self.config.get('params', {}))
        self.build_pipeline()

class VotingEnsembleClassifier(BaselineClassifier):
    """Voting Ensemble for connectome classification"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # The 'estimators' parameter should be a list of (name, estimator) tuples
        # Estimators themselves should be scikit-learn compatible models, not BaselineClassifier instances
        estimators_config = self.config.get('estimators', [])
        estimators = []
        for name, model_type, params in estimators_config:
            if model_type in MODEL_CLASSES:
                estimators.append((name, MODEL_CLASSES[model_type](**params)))
            else:
                raise ValueError(f"Unknown model type for VotingClassifier: {model_type}")

        self.model = VotingClassifier(estimators=estimators, voting=self.config.get('voting', 'hard'), **self.config.get('params', {}))
        self.build_pipeline()

class BaggingEnsembleClassifier(BaselineClassifier):
    """Bagging Ensemble for connectome classification"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # The 'base_estimator' parameter should be a scikit-learn compatible model, not a BaselineClassifier instance
        base_estimator_config = self.config.get('base_estimator', {})
        model_type = base_estimator_config.get('model_type', 'SVC')
        params = base_estimator_config.get('params', {})
        if model_type in MODEL_CLASSES:
            base_estimator = MODEL_CLASSES[model_type](**params)
        else:
            raise ValueError(f"Unknown model type for BaggingClassifier base estimator: {model_type}")

        self.model = BaggingClassifier(base_estimator=base_estimator, **self.config.get('params', {}))
        self.build_pipeline()


# Dictionary to map classifier names to their classes
BASELINE_CLASSIFIER_CLASSES = {
    'SVMClassifier': SVMClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'LogisticRegressionClassifier': LogisticRegressionClassifier,
    'VotingEnsembleClassifier': VotingEnsembleClassifier,
    'BaggingEnsembleClassifier': BaggingEnsembleClassifier
}

def create_baseline_classifier(model_name: str, config: Dict[str, Any]) -> BaselineClassifier:
    """
    Factory function to create a baseline classifier instance.

    Args:
        model_name: The name of the baseline classifier to create (e.g., 'SVMClassifier').
        config: A dictionary containing configuration parameters for the classifier.

    Returns:
        An instance of a BaselineClassifier subclass.

    Raises:
        ValueError: If an unknown model name is provided.
    """
    if model_name not in BASELINE_CLASSIFIER_CLASSES:
        raise ValueError(f"Unknown baseline classifier: {model_name}. "
                         f"Available classifiers are: {list(BASELINE_CLASSIFIER_CLASSES.keys())}")
    return BASELINE_CLASSIFIER_CLASSES[model_name](config)

# Helper function to flatten connectivity matrix (upper triangle)
def flatten_connectivity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Flattens the upper triangular part of a connectivity matrix.

    Args:
        matrix: A square numpy array representing the connectivity matrix.

    Returns:
        A 1D numpy array containing the flattened upper triangular elements.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    # Get upper triangle without diagonal
    upper_triangle_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_triangle_indices]

# Example usage (requires data loading and preparation)
if __name__ == '__main__':
    print("Baseline classifier module initialized.")
    print("Requires data loading and training pipeline for full functionality.")

    # Example of how to use the factory function with dummy data
    try:
        # Dummy data: 100 subjects, 116x116 connectivity matrices
        num_subjects = 100
        num_regions = 116
        dummy_X_matrices = np.random.rand(num_subjects, num_regions, num_regions)
        dummy_y = np.random.randint(0, 2, num_subjects) # Binary labels

        # Flatten matrices
        dummy_X_flat = np.array([flatten_connectivity_matrix(matrix) for matrix in dummy_X_matrices])

        # Example config for SVM
        svm_config = {
            'params': {'C': 1.0, 'kernel': 'linear'},
            'feature_selection': {'method': 'select_k_best', 'k': 50} # Reduced k for dummy data
        }
        # Create SVM model using the factory function
        svm_model = create_baseline_classifier('SVMClassifier', svm_config)

        # Train and evaluate (using dummy data - cross-validation would be in trainer.py)
        svm_model.train(dummy_X_flat, dummy_y)
        metrics = svm_model.evaluate(dummy_X_flat, dummy_y)
        print(f"Dummy SVM Metrics: {metrics}")

        # Example config for RandomForest
        rf_config = {
            'params': {'n_estimators': 100, 'random_state': 42},
            'feature_selection': {'method': 'variance_threshold', 'threshold': 0.01}
        }
        # Create RandomForest model using the factory function
        rf_model = create_baseline_classifier('RandomForestClassifier', rf_config)
        rf_model.train(dummy_X_flat, dummy_y)
        metrics_rf = rf_model.evaluate(dummy_X_flat, dummy_y)
        print(f"Dummy RandomForest Metrics: {metrics_rf}")

    except Exception as e:
        print(f"Error during dummy example execution: {e}")
        print("This is expected as data loading is not implemented here.")
