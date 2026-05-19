
from typing import List, Any, Dict, Optional
from algorithms.base import Algorithm
import numpy as np

class KMeansClustering(Algorithm):
    name = "kmeans"
    tags = ["ml", "clustering", "unsupervised", "iterative"]
    complexity = {"time": "O(n*k*i)", "space": "O(n+k)"}
    performance_profiles = {
        "clustering": {"score": 0.85, "conditions": {"task": "clustering"}},
        "general": {"score": 0.6, "conditions": {}}
    }

    def __init__(self):
        super().__init__()
        self.set_params(n_clusters=3, max_iter=100, random_state=42)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, dict):
            if "labels" not in output_data and "centers" not in output_data:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            from sklearn.cluster import KMeans
            n_clusters = self.params.get("n_clusters", 3)
            max_iter = self.params.get("max_iter", 100)
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            return {
                "labels": labels,
                "centers": kmeans.cluster_centers_,
                "inertia": kmeans.inertia_
            }
        return data

class DBSCANClustering(Algorithm):
    name = "dbscan"
    tags = ["ml", "clustering", "density", "noise_detection"]
    complexity = {"time": "O(n log n)", "space": "O(n)"}
    performance_profiles = {
        "density_clustering": {"score": 0.9, "conditions": {"cluster_shape": "irregular"}},
        "general": {"score": 0.7, "conditions": {}}
    }

    def __init__(self):
        super().__init__()
        self.set_params(eps=0.5, min_samples=5)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, dict):
            if "labels" not in output_data:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            from sklearn.cluster import DBSCAN
            eps = self.params.get("eps", 0.5)
            min_samples = self.params.get("min_samples", 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            return {
                "labels": labels,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "n_noise": list(labels).count(-1)
            }
        return data

class RandomForestClassifier(Algorithm):
    name = "random_forest"
    tags = ["ml", "classification", "supervised", "ensemble"]
    complexity = {"time": "O(n*m*log(n))", "space": "O(n*m)"}
    performance_profiles = {
        "classification": {"score": 0.9, "conditions": {"task": "classification"}},
        "general": {"score": 0.7, "conditions": {}}
    }

    def __init__(self):
        super().__init__()
        self.set_params(n_estimators=100, max_depth=10)
        self._model = None

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, dict):
            if not output_data.get("trained", False):
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, dict) and "X_train" in data and "y_train" in data:
            from sklearn.ensemble import RandomForestClassifier as RFC
            n_estimators = self.params.get("n_estimators", 100)
            max_depth = self.params.get("max_depth", 10)

            self._model = RFC(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self._model.fit(data["X_train"], data["y_train"])

            result = {"trained": True}
            if "X_test" in data:
                result["predictions"] = self._model.predict(data["X_test"])
                result[" probabilities"] = self._model.predict_proba(data["X_test"]).tolist()

            return result
        return data

class LinearRegression(Algorithm):
    name = "linear_regression"
    tags = ["ml", "regression", "supervised", "fast"]
    complexity = {"time": "O(n*d²)", "space": "O(d²)"}
    performance_profiles = {
        "regression": {"score": 0.85, "conditions": {"task": "regression"}},
        "general": {"score": 0.6, "conditions": {}}
    }

    def __init__(self):
        super().__init__()
        self.set_params(fit_intercept=True)
        self._model = None

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, dict):
            if not output_data.get("trained", False):
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, dict) and "X_train" in data and "y_train" in data:
            from sklearn.linear_model import LinearRegression as LR
            fit_intercept = self.params.get("fit_intercept", True)

            self._model = LR(fit_intercept=fit_intercept)
            self._model.fit(data["X_train"], data["y_train"])

            result = {
                "trained": True,
                "coef": self._model.coef_.tolist(),
                "intercept": float(self._model.intercept_)
            }

            if "X_test" in data:
                result["predictions"] = self._model.predict(data["X_test"]).tolist()

            return result
        return data
