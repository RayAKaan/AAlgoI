
"""
algorithms/ml/__init__.py

ML/DL algorithm registry. All algorithms use the MLAlgorithm base class.

Categories:
- Regression: linear_regression, ridge, lasso
- Classification: logistic_regression, knn, svm, gaussian_nb
- Ensemble: random_forest, xgboost, lightgbm
- Clustering: gmm, kmeans, dbscan
- Dimensionality Reduction: pca
"""

from typing import Any

from aalgoi.algorithms.ml.base import MLAlgorithm
from aalgoi.algorithms.ml.classical import (
    GaussianNBAlgo,
    KNNAlgo,
    LassoAlgo,
    LinearRegressionAlgo,
    LogisticRegressionAlgo,
    RidgeAlgo,
    SVMAlgo,
)
from aalgoi.algorithms.ml.clustering import (
    GMMAlgo,
    PCAReductionAlgo,
)
from aalgoi.algorithms.ml.ensemble import (
    LightGBMAlgo,
    RandomForestAlgo,
    XGBoostAlgo,
)


class KMeansClustering(MLAlgorithm):
    """K-Means clustering."""

    def __init__(self) -> None:
        from sklearn.cluster import KMeans
        super().__init__(
            KMeans,
            {"n_clusters": 3, "random_state": 42, "n_init": 10},
            name="kmeans",
            task="clustering"
        )
        self.tags = ["ml", "clustering", "centroid", "unsupervised"]
        self.best_for = ["spherical_clusters", "large_data", "fast"]
        self.time_complexity = "O(n*k*i)"
        self.space_complexity = "O(k*d)"
        self.patterns = ["CentroidBased", "Partitioning"]
        self.problem_types = ["CLUSTERING"]

    def process(self, data: Any) -> Any:
        X_train, _, _ = self._extract_data(data)
        if X_train is None:
            return {"trained": False, "error": "invalid data"}

        try:
            self.model = self.model_class(**self.default_params)
            labels = self.model.fit_predict(X_train)
            return {
                "trained": True,
                "algorithm": self._name,
                "labels": labels.tolist(),
                "n_clusters": self.model.n_clusters,
                "cluster_centers": self.model.cluster_centers_.tolist(),
            }
        except Exception as e:
            return {"trained": False, "error": str(e)}


class DBSCANClustering(MLAlgorithm):
    """DBSCAN density-based clustering."""

    def __init__(self) -> None:
        from sklearn.cluster import DBSCAN
        super().__init__(
            DBSCAN,
            {"eps": 0.5, "min_samples": 5},
            name="dbscan",
            task="clustering"
        )
        self.tags = ["ml", "clustering", "density", "unsupervised", "noise_detection"]
        self.best_for = ["arbitrary_shapes", "noise_detection", "variable_density"]
        self.time_complexity = "O(n*log(n))"
        self.space_complexity = "O(n)"
        self.patterns = ["DensityBased", "NoiseDetection"]
        self.problem_types = ["CLUSTERING"]

    def process(self, data: Any) -> Any:
        X_train, _, _ = self._extract_data(data)
        if X_train is None:
            return {"trained": False, "error": "invalid data"}

        try:
            self.model = self.model_class(**self.default_params)
            labels = self.model.fit_predict(X_train)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            return {
                "trained": True,
                "algorithm": self._name,
                "labels": labels.tolist(),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
            }
        except Exception as e:
            return {"trained": False, "error": str(e)}


LinearRegression = LinearRegressionAlgo
RandomForestClassifier = RandomForestAlgo


__all__ = [
    "MLAlgorithm",
    "LinearRegressionAlgo",
    "RidgeAlgo",
    "LassoAlgo",
    "LogisticRegressionAlgo",
    "KNNAlgo",
    "SVMAlgo",
    "GaussianNBAlgo",
    "RandomForestAlgo",
    "XGBoostAlgo",
    "LightGBMAlgo",
    "KMeansClustering",
    "DBSCANClustering",
    "GMMAlgo",
    "PCAReductionAlgo",
    "LinearRegression",
    "RandomForestClassifier",
]
