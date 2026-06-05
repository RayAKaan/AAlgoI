
from aalgoi.algorithms.ml.base import MLAlgorithm
import numpy as np


class GMMAlgo(MLAlgorithm):
    """Gaussian Mixture Model for probabilistic clustering."""

    def __init__(self):
        from sklearn.mixture import GaussianMixture
        super().__init__(
            GaussianMixture,
            {"n_components": 3, "random_state": 42},
            name="gmm",
            task="clustering"
        )
        self.tags = ["ml", "clustering", "probabilistic", "soft", "gaussian"]
        self.best_for = ["density_estimation", "soft_clustering", "elliptical_clusters"]
        self.time_complexity = "O(n*k*d)"
        self.space_complexity = "O(k*d)"
        self.patterns = ["Probabilistic", "SoftClustering"]
        self.problem_types = ["CLUSTERING"]

    def process(self, data):
        X_train, y_train, X_test = self._extract_data(data)

        if X_train is None:
            return {"trained": False, "error": "invalid data format"}

        try:
            self.model = self.model_class(**self.default_params)
            labels = self.model.fit_predict(X_train)

            result = {
                "trained": True,
                "algorithm": self._name,
                "labels": labels.tolist(),
                "n_clusters": self.model.n_components,
            }

            if hasattr(self.model, 'weights_'):
                result["weights"] = self.model.weights_.tolist()

            if X_test is not None:
                result["predictions"] = self.model.predict(X_test).tolist()
                if hasattr(self.model, 'predict_proba'):
                    result["probabilities"] = self.model.predict_proba(X_test).tolist()

            return result

        except Exception as e:
            return {"trained": False, "error": str(e)}


class PCAReductionAlgo(MLAlgorithm):
    """Principal Component Analysis for dimensionality reduction."""

    def __init__(self):
        from sklearn.decomposition import PCA
        super().__init__(
            PCA,
            {"n_components": 2, "random_state": 42},
            name="pca",
            task="dimensionality_reduction"
        )
        self.tags = ["ml", "dimensionality_reduction", "linear", "unsupervised", "reduce", "dimensions"]
        self.best_for = ["visualization", "noise_reduction", "feature_extraction", "reduce_dimensions"]
        self.time_complexity = "O(n*d\u00b2)"
        self.space_complexity = "O(d\u00b2)"
        self.patterns = ["LinearReduction", "Unsupervised", "t-SNE", "Visualization"]
        self.problem_types = ["DIMENSIONALITY_REDUCTION"]

    def process(self, data):
        X_train, y_train, X_test = self._extract_data(data)

        if X_train is None:
            return {"trained": False, "error": "invalid data format"}

        try:
            self.model = self.model_class(**self.default_params)
            transformed = self.model.fit_transform(X_train)

            result = {
                "trained": True,
                "algorithm": self._name,
                "transformed": transformed.tolist(),
                "n_components": self.model.n_components_,
                "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            }

            if X_test is not None:
                result["predictions"] = self.model.transform(X_test).tolist()

            return result

        except Exception as e:
            return {"trained": False, "error": str(e)}
