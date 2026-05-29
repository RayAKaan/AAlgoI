
import numpy as np
from algorithms.ml.base import MLAlgorithm
import logging

logger = logging.getLogger(__name__)


class LinearRegressionAlgo(MLAlgorithm):
    """Ordinary least squares Linear Regression."""

    def __init__(self):
        from sklearn.linear_model import LinearRegression
        super().__init__(
            LinearRegression,
            name="linear_regression",
            task="regression"
        )
        self.tags = ["ml", "regression", "supervised", "linear", "fast"]
        self.best_for = ["small_data", "linear_relationship", "baseline"]
        self.time_complexity = "O(n*d\u00b2)"
        self.space_complexity = "O(d\u00b2)"
        self.patterns = ["LinearModel", "Parametric"]
        self.problem_types = ["REGRESSION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            r2 = float(self.model.score(X, y))
            return {"r2_score": r2}
        except Exception:
            return {}


class RidgeAlgo(MLAlgorithm):
    """Linear least squares with l2 regularization."""

    def __init__(self):
        from sklearn.linear_model import Ridge
        super().__init__(
            Ridge,
            {"alpha": 1.0},
            name="ridge",
            task="regression"
        )
        self.tags = ["ml", "regression", "regularized", "linear"]
        self.best_for = ["multicollinearity", "overfitting_prevention"]
        self.time_complexity = "O(n*d\u00b2)"
        self.space_complexity = "O(d\u00b2)"
        self.patterns = ["LinearModel", "Regularized"]
        self.problem_types = ["REGRESSION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            r2 = float(self.model.score(X, y))
            return {"r2_score": r2}
        except Exception:
            return {}


class LassoAlgo(MLAlgorithm):
    """Linear Model trained with L1 prior as regularizer."""

    def __init__(self):
        from sklearn.linear_model import Lasso
        super().__init__(
            Lasso,
            {"alpha": 1.0, "max_iter": 1000},
            name="lasso",
            task="regression"
        )
        self.tags = ["ml", "regression", "regularized", "feature_selection"]
        self.best_for = ["feature_selection", "sparse_solutions"]
        self.time_complexity = "O(n*d\u00b2)"
        self.space_complexity = "O(d\u00b2)"
        self.patterns = ["LinearModel", "Regularized", "FeatureSelection"]
        self.problem_types = ["REGRESSION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            r2 = float(self.model.score(X, y))
            n_used = int(np.sum(self.model.coef_ != 0)) if hasattr(self.model, 'coef_') else 0
            return {"r2_score": r2, "n_features_used": n_used}
        except Exception:
            return {}


class LogisticRegressionAlgo(MLAlgorithm):
    """Logistic Regression (aka logit, MaxEnt) classifier."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        super().__init__(
            LogisticRegression,
            {"max_iter": 1000, "solver": "lbfgs"},
            name="logistic_regression",
            task="classification"
        )
        self.tags = ["ml", "classification", "linear", "probabilistic"]
        self.best_for = ["binary_classification", "probability_output", "baseline"]
        self.time_complexity = "O(n*d)"
        self.space_complexity = "O(d)"
        self.patterns = ["LinearModel", "Probabilistic"]
        self.problem_types = ["CLASSIFICATION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            from sklearn.metrics import accuracy_score
            preds = self.model.predict(X)
            acc = float(accuracy_score(y, preds))
            return {"accuracy": acc}
        except Exception:
            return {}


class KNNAlgo(MLAlgorithm):
    """Classifier implementing the k-nearest neighbors vote."""

    def __init__(self):
        from sklearn.neighbors import KNeighborsClassifier
        super().__init__(
            KNeighborsClassifier,
            {"n_neighbors": 5},
            name="knn",
            task="classification"
        )
        self.tags = ["ml", "classification", "distance", "nonparametric"]
        self.best_for = ["small_data", "decision_boundaries", "prototype"]
        self.time_complexity = "O(n*d)"
        self.space_complexity = "O(n*d)"
        self.patterns = ["DistanceBased", "NonParametric"]
        self.problem_types = ["CLASSIFICATION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            from sklearn.metrics import accuracy_score
            preds = self.model.predict(X)
            acc = float(accuracy_score(y, preds))
            return {"accuracy": acc}
        except Exception:
            return {}


class SVMAlgo(MLAlgorithm):
    """C-Support Vector Classification."""

    def __init__(self):
        from sklearn.svm import SVC
        super().__init__(
            SVC,
            {"kernel": "rbf", "probability": True, "C": 1.0},
            name="svm",
            task="classification"
        )
        self.tags = ["ml", "classification", "kernel", "margin"]
        self.best_for = ["high_dim", "clear_margin", "small_medium_data"]
        self.time_complexity = "O(n\u00b2*d)"
        self.space_complexity = "O(n\u00b2)"
        self.patterns = ["KernelMethod", "MarginBased"]
        self.problem_types = ["CLASSIFICATION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            from sklearn.metrics import accuracy_score
            preds = self.model.predict(X)
            acc = float(accuracy_score(y, preds))
            n_support = len(self.model.support_) if hasattr(self.model, 'support_') else 0
            return {"accuracy": acc, "n_support": n_support}
        except Exception:
            return {}


class GaussianNBAlgo(MLAlgorithm):
    """Gaussian Naive Bayes."""

    def __init__(self):
        from sklearn.naive_bayes import GaussianNB
        super().__init__(
            GaussianNB,
            name="gaussian_nb",
            task="classification"
        )
        self.tags = ["ml", "classification", "probabilistic", "fast", "naive_bayes"]
        self.best_for = ["high_dim", "real_time", "small_data", "text_baseline"]
        self.time_complexity = "O(n*d)"
        self.space_complexity = "O(d)"
        self.patterns = ["Probabilistic", "NaiveBayes"]
        self.problem_types = ["CLASSIFICATION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            from sklearn.metrics import accuracy_score
            preds = self.model.predict(X)
            acc = float(accuracy_score(y, preds))
            return {"accuracy": acc}
        except Exception:
            return {}
