from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="knn_classifier",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.ML,
    complexity=Complexity("O(n*d)", "O(n*d)", "n*d", "n*d"),
    principles=frozenset({"lazy_learning"}),
    deterministic=False, exact=False,
    tags=frozenset({"ml", "optional"}),
))
class KNNClassifier(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from sklearn.neighbors import KNeighborsClassifier
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train = spec.inputs.get("X_train", spec.inputs.get("train_x", []))
        y_train = spec.inputs.get("y_train", spec.inputs.get("train_y", []))
        X_test = spec.inputs.get("X_test", spec.inputs.get("test_x", X_train))
        if not X_train or not y_train:
            return []
        n = min(len(X_train), max(1, int(np.sqrt(len(X_train)))))
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(np.array(X_train), np.array(y_train))
        return knn.predict(np.array(X_test)).tolist()


@algorithm(AlgorithmSpec(
    name="linear_regression",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.ML,
    complexity=Complexity("O(n*d^2)", "O(d^2)", "n*d^2", "d^2"),
    principles=frozenset({"least_squares"}),
    deterministic=True, exact=False,
    tags=frozenset({"ml", "optional"}),
))
class LinearRegressionAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train = spec.inputs.get("X_train", spec.inputs.get("train_x", []))
        y_train = spec.inputs.get("y_train", spec.inputs.get("train_y", []))
        X_test = spec.inputs.get("X_test", spec.inputs.get("test_x", X_train))
        if not X_train or not y_train:
            return []
        lr = LinearRegression()
        lr.fit(np.array(X_train), np.array(y_train))
        return lr.predict(np.array(X_test)).tolist()


@algorithm(AlgorithmSpec(
    name="kmeans",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.ML,
    complexity=Complexity("O(n*k*d)", "O(n+k*d)", "n*k*d", "n+k*d"),
    principles=frozenset({"expectation_maximization"}),
    deterministic=False, exact=False,
    tags=frozenset({"ml", "optional"}),
))
class KMeans(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        data = spec.inputs.get("data", spec.inputs.get("X", []))
        if not data:
            return {"labels": [], "centers": [], "n_clusters": 0}
        X = np.array(data)
        n = min(8, max(1, len(X) // 2))
        km = KMeans(n_clusters=n, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        return {
            "labels": labels.tolist(),
            "centers": km.cluster_centers_.tolist(),
            "n_clusters": n,
        }
