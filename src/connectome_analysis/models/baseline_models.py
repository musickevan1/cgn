import numpy as np
from typing import Literal
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class BaselineClassifier:
    """
    Base class for baseline classifiers.
    """
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        if y_prob is not None and y_prob.shape[1] == 2: # For binary classification
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        return metrics

    def predict_proba(self, X_test):
        raise NotImplementedError

class SVMClassifier(BaselineClassifier):
    """
    SVM Classifier for connectome data.
    """
    def __init__(self, C: float = 1.0, kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf', random_state: int = 42):
        super().__init__()
        self.model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

class RandomForest(BaselineClassifier):
    """
    Random Forest Classifier for connectome data.
    """
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

class LogisticRegressionClassifier(BaselineClassifier):
    """
    Logistic Regression Classifier for connectome data.
    """
    def __init__(self, solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] = 'liblinear', random_state: int = 42):
        super().__init__()
        self.model = LogisticRegression(solver=solver, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

# Example Usage (for testing purposes, will be removed or moved later)
if __name__ == "__main__":
    # Generate some dummy data
    X = np.random.rand(100, 50) # 100 samples, 50 features
    y = np.random.randint(0, 2, 100) # Binary labels

    # Split data (simple split for example)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    print("--- SVM Classifier ---")
    svm_clf = SVMClassifier()
    svm_clf.train(X_train, y_train)
    svm_metrics = svm_clf.evaluate(X_test, y_test)
    print(f"SVM Metrics: {svm_metrics}")

    print("\n--- Random Forest Classifier ---")
    rf_clf = RandomForest()
    rf_clf.train(X_train, y_train)
    rf_metrics = rf_clf.evaluate(X_test, y_test)
    print(f"Random Forest Metrics: {rf_metrics}")

    print("\n--- Logistic Regression Classifier ---")
    lr_clf = LogisticRegressionClassifier()
    lr_clf.train(X_train, y_train)
    lr_metrics = lr_clf.evaluate(X_test, y_test)
    print(f"Logistic Regression Metrics: {lr_metrics}")
