from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


def _require(module_path: str):
    parts = module_path.split(".")
    mod = __import__(parts[0])
    for p in parts[1:]:
        mod = getattr(mod, p)
    return mod


def _train_test(spec: ProblemSpec) -> tuple[Any, Any, Any]:
    X_train = spec.inputs.get("X_train", spec.inputs.get("train_x", []))
    y_train = spec.inputs.get("y_train", spec.inputs.get("train_y", []))
    X_test = spec.inputs.get("X_test", spec.inputs.get("test_x", X_train))
    return X_train, y_train, X_test


def _result_list(arr: Any) -> list:
    return arr.tolist() if hasattr(arr, "tolist") else list(arr)


@algorithm(AlgorithmSpec(
    name="knn_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d)", "O(n*d)", "n*d", "n*d"),
    principles=frozenset({"lazy_learning"}),
    deterministic=False, exact=False,
    tags=frozenset({"classification", "ml", "optional"}),
))
class KNNClassifier(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from sklearn.neighbors import KNeighborsClassifier
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        n = min(len(X_train), max(1, int(np.sqrt(len(X_train)))))
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(np.array(X_train), np.array(y_train))
        return _result_list(knn.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="logistic_regression",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d)", "O(d)", "n*d", "d"),
    principles=frozenset({"linear_model", "logistic"}),
    deterministic=True, exact=False,
    tags=frozenset({"classification", "ml", "optional"}),
))
class LogisticRegressionAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            LogisticRegression = _require("sklearn.linear_model.LogisticRegression")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = LogisticRegression(max_iter=int(spec.inputs.get("max_iter", 1000)))
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="decision_tree_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d*log n)", "O(nodes)", "n*d*log n", "nodes"),
    principles=frozenset({"decision_tree", "information_gain"}),
    deterministic=True, exact=False,
    tags=frozenset({"classification", "tree", "ml", "optional"}),
))
class DecisionTreeClassifierAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            DecisionTreeClassifier = _require("sklearn.tree.DecisionTreeClassifier")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = DecisionTreeClassifier(random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="random_forest_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(trees*n*d*log n)", "O(trees*nodes)", "trees*n*d*log n", "trees*nodes"),
    principles=frozenset({"ensemble", "bagging"}),
    deterministic=False, exact=False,
    tags=frozenset({"classification", "ensemble", "ml", "optional"}),
))
class RandomForestClassifierAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            RandomForestClassifier = _require("sklearn.ensemble.RandomForestClassifier")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = RandomForestClassifier(n_estimators=int(spec.inputs.get("n_estimators", 100)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="svm_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(n^2*d)", "O(n*d)", "n^2*d", "n*d"),
    principles=frozenset({"maximum_margin", "kernel_trick"}),
    deterministic=True, exact=False,
    tags=frozenset({"classification", "svm", "ml", "optional"}),
))
class SVMClassifierAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            SVC = _require("sklearn.svm.SVC")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = SVC(kernel=spec.inputs.get("kernel", "rbf"), probability=False, random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="naive_bayes_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d)", "O(d)", "n*d", "d"),
    principles=frozenset({"bayesian", "conditional_independence"}),
    deterministic=True, exact=False,
    tags=frozenset({"classification", "bayesian", "ml", "optional"}),
))
class NaiveBayesClassifierAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            GaussianNB = _require("sklearn.naive_bayes.GaussianNB")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = GaussianNB()
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="gradient_boosting_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(trees*n*d*log n)", "O(trees*nodes)", "trees*n*d*log n", "trees*nodes"),
    principles=frozenset({"ensemble", "boosting"}),
    deterministic=False, exact=False,
    tags=frozenset({"classification", "ensemble", "ml", "optional"}),
))
class GradientBoostingClassifierAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            GradientBoostingClassifier = _require("sklearn.ensemble.GradientBoostingClassifier")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = GradientBoostingClassifier(n_estimators=int(spec.inputs.get("n_estimators", 100)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="extra_trees_classifier",
    task=ProblemTask.CLASSIFICATION,
    domain=Domain.ML,
    complexity=Complexity("O(trees*n*d*log n)", "O(trees*nodes)", "trees*n*d*log n", "trees*nodes"),
    principles=frozenset({"ensemble", "randomization"}),
    deterministic=False, exact=False,
    tags=frozenset({"classification", "ensemble", "ml", "optional"}),
))
class ExtraTreesClassifierAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            ExtraTreesClassifier = _require("sklearn.ensemble.ExtraTreesClassifier")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = ExtraTreesClassifier(n_estimators=int(spec.inputs.get("n_estimators", 100)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="linear_regression",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d^2)", "O(d^2)", "n*d^2", "d^2"),
    principles=frozenset({"least_squares"}),
    deterministic=True, exact=False,
    tags=frozenset({"regression", "ml", "optional"}),
))
class LinearRegressionAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        lr = LinearRegression()
        lr.fit(np.array(X_train), np.array(y_train))
        return _result_list(lr.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="ridge_regression",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d^2)", "O(d^2)", "n*d^2", "d^2"),
    principles=frozenset({"regularized_least_squares", "l2"}),
    deterministic=True, exact=False,
    tags=frozenset({"regression", "regularized", "ml", "optional"}),
))
class RidgeRegressionAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            Ridge = _require("sklearn.linear_model.Ridge")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = Ridge(alpha=float(spec.inputs.get("alpha", 1.0)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="lasso_regression",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d^2)", "O(d^2)", "n*d^2", "d^2"),
    principles=frozenset({"regularized_least_squares", "l1"}),
    deterministic=True, exact=False,
    tags=frozenset({"regression", "regularized", "ml", "optional"}),
))
class LassoRegressionAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            Lasso = _require("sklearn.linear_model.Lasso")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = Lasso(alpha=float(spec.inputs.get("alpha", 1.0)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="decision_tree_regressor",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d*log n)", "O(nodes)", "n*d*log n", "nodes"),
    principles=frozenset({"decision_tree", "regression"}),
    deterministic=True, exact=False,
    tags=frozenset({"regression", "tree", "ml", "optional"}),
))
class DecisionTreeRegressorAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            DecisionTreeRegressor = _require("sklearn.tree.DecisionTreeRegressor")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = DecisionTreeRegressor(random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="random_forest_regressor",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(trees*n*d*log n)", "O(trees*nodes)", "trees*n*d*log n", "trees*nodes"),
    principles=frozenset({"ensemble", "bagging"}),
    deterministic=False, exact=False,
    tags=frozenset({"regression", "ensemble", "ml", "optional"}),
))
class RandomForestRegressorAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            RandomForestRegressor = _require("sklearn.ensemble.RandomForestRegressor")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = RandomForestRegressor(n_estimators=int(spec.inputs.get("n_estimators", 100)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="svr_regressor",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(n^2*d)", "O(n*d)", "n^2*d", "n*d"),
    principles=frozenset({"maximum_margin", "kernel_trick"}),
    deterministic=True, exact=False,
    tags=frozenset({"regression", "svm", "ml", "optional"}),
))
class SVRRegressorAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            SVR = _require("sklearn.svm.SVR")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = SVR(kernel=spec.inputs.get("kernel", "rbf"))
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="gradient_boosting_regressor",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(trees*n*d*log n)", "O(trees*nodes)", "trees*n*d*log n", "trees*nodes"),
    principles=frozenset({"ensemble", "boosting"}),
    deterministic=False, exact=False,
    tags=frozenset({"regression", "ensemble", "ml", "optional"}),
))
class GradientBoostingRegressorAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            GradientBoostingRegressor = _require("sklearn.ensemble.GradientBoostingRegressor")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = GradientBoostingRegressor(n_estimators=int(spec.inputs.get("n_estimators", 100)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="extra_trees_regressor",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(trees*n*d*log n)", "O(trees*nodes)", "trees*n*d*log n", "trees*nodes"),
    principles=frozenset({"ensemble", "randomization"}),
    deterministic=False, exact=False,
    tags=frozenset({"regression", "ensemble", "ml", "optional"}),
))
class ExtraTreesRegressorAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            ExtraTreesRegressor = _require("sklearn.ensemble.ExtraTreesRegressor")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        model = ExtraTreesRegressor(n_estimators=int(spec.inputs.get("n_estimators", 100)), random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="polynomial_regression",
    task=ProblemTask.REGRESSION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d^degree)", "O(d^degree)", "n*d^degree", "d^degree"),
    principles=frozenset({"feature_expansion", "least_squares"}),
    deterministic=True, exact=False,
    tags=frozenset({"regression", "polynomial", "ml", "optional"}),
))
class PolynomialRegressionAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            PolynomialFeatures = _require("sklearn.preprocessing.PolynomialFeatures")
            LinearRegression = _require("sklearn.linear_model.LinearRegression")
            from sklearn.pipeline import make_pipeline
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        X_train, y_train, X_test = _train_test(spec)
        if not X_train or not y_train:
            return []
        degree = int(spec.inputs.get("degree", 2))
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        model.fit(np.array(X_train), np.array(y_train))
        return _result_list(model.predict(np.array(X_test)))


@algorithm(AlgorithmSpec(
    name="kmeans",
    task=ProblemTask.CLUSTERING,
    domain=Domain.ML,
    complexity=Complexity("O(n*k*d)", "O(n+k*d)", "n*k*d", "n+k*d"),
    principles=frozenset({"expectation_maximization"}),
    deterministic=False, exact=False,
    tags=frozenset({"clustering", "ml", "optional"}),
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
        n = min(int(spec.inputs.get("n_clusters", 8)), max(1, len(X) // 2))
        km = KMeans(n_clusters=n, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        return {
            "labels": labels.tolist(),
            "centers": km.cluster_centers_.tolist(),
            "n_clusters": n,
        }


@algorithm(AlgorithmSpec(
    name="dbscan",
    task=ProblemTask.CLUSTERING,
    domain=Domain.ML,
    complexity=Complexity("O(n*d)", "O(n*d)", "n*d", "n*d"),
    principles=frozenset({"density_based", "noise"}),
    deterministic=True, exact=False,
    tags=frozenset({"clustering", "density", "ml", "optional"}),
))
class DBSCAN(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            DBSCAN = _require("sklearn.cluster.DBSCAN")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        data = spec.inputs.get("data", spec.inputs.get("X", []))
        if not data:
            return {"labels": [], "n_clusters": 0}
        eps = float(spec.inputs.get("eps", 0.5))
        min_samples = int(spec.inputs.get("min_samples", 2))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(np.array(data))
        n_clusters = len(set(labels) - {-1})
        return {"labels": labels.tolist(), "n_clusters": n_clusters}


@algorithm(AlgorithmSpec(
    name="agglomerative_clustering",
    task=ProblemTask.CLUSTERING,
    domain=Domain.ML,
    complexity=Complexity("O(n^3)", "O(n^2)", "n^3", "n^2"),
    principles=frozenset({"hierarchical", "linkage"}),
    deterministic=True, exact=False,
    tags=frozenset({"clustering", "hierarchical", "ml", "optional"}),
))
class AgglomerativeClusteringAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            AgglomerativeClustering = _require("sklearn.cluster.AgglomerativeClustering")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        data = spec.inputs.get("data", spec.inputs.get("X", []))
        if not data:
            return {"labels": [], "n_clusters": 0}
        n = min(int(spec.inputs.get("n_clusters", 8)), max(1, len(data) // 2))
        model = AgglomerativeClustering(n_clusters=n)
        labels = model.fit_predict(np.array(data))
        return {"labels": labels.tolist(), "n_clusters": n}


@algorithm(AlgorithmSpec(
    name="gaussian_mixture",
    task=ProblemTask.CLUSTERING,
    domain=Domain.ML,
    complexity=Complexity("O(n*k*d^2)", "O(n*d)", "n*k*d^2", "n*d"),
    principles=frozenset({"expectation_maximization", "gaussian"}),
    deterministic=False, exact=False,
    tags=frozenset({"clustering", "gaussian", "ml", "optional"}),
))
class GaussianMixtureAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            GaussianMixture = _require("sklearn.mixture.GaussianMixture")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        data = spec.inputs.get("data", spec.inputs.get("X", []))
        if not data:
            return {"labels": [], "n_clusters": 0}
        n = min(int(spec.inputs.get("n_clusters", 8)), max(1, len(data) // 2))
        model = GaussianMixture(n_components=n, random_state=42)
        labels = model.fit_predict(np.array(data))
        return {"labels": labels.tolist(), "n_clusters": n}


@algorithm(AlgorithmSpec(
    name="pca",
    task=ProblemTask.DIMENSIONALITY_REDUCTION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d^2)", "O(d^2)", "n*d^2", "d^2"),
    principles=frozenset({"variance_maximization", "orthogonal"}),
    deterministic=True, exact=False,
    tags=frozenset({"dimensionality_reduction", "ml", "optional"}),
))
class PCA(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            PCA = _require("sklearn.decomposition.PCA")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        data = spec.inputs.get("data", spec.inputs.get("X", []))
        if not data:
            return {"transformed": [], "components": [], "explained_variance_ratio": []}
        n = int(spec.inputs.get("n_components", min(2, len(np.array(data)[0]) if len(np.array(data).shape) > 1 else 1)))
        model = PCA(n_components=n)
        transformed = model.fit_transform(np.array(data))
        return {
            "transformed": transformed.tolist(),
            "components": model.components_.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        }


@algorithm(AlgorithmSpec(
    name="isolation_forest",
    task=ProblemTask.ANOMALY_DETECTION,
    domain=Domain.ML,
    complexity=Complexity("O(n*d)", "O(n*d)", "n*d", "n*d"),
    principles=frozenset({"ensemble", "isolation"}),
    deterministic=False, exact=False,
    tags=frozenset({"anomaly_detection", "ml", "optional"}),
))
class IsolationForestAlgo(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            IsolationForest = _require("sklearn.ensemble.IsolationForest")
        except ImportError:
            raise OptionalDependencyMissing("ml", "scikit-learn")
        data = spec.inputs.get("data", spec.inputs.get("X", []))
        X_test = spec.inputs.get("X_test", spec.inputs.get("test_x", data))
        if not data:
            return {"labels": [], "scores": []}
        model = IsolationForest(contamination=float(spec.inputs.get("contamination", 0.1)), random_state=42)
        model.fit(np.array(data))
        labels = model.predict(np.array(X_test))
        scores = model.decision_function(np.array(X_test))
        return {
            "labels": [1 if l == -1 else 0 for l in labels.tolist()],
            "scores": scores.tolist(),
        }
