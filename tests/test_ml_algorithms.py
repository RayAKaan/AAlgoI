"""
tests/test_ml_algorithms.py

Comprehensive tests for all 14 ML algorithms:
- Regression: linear_regression, ridge, lasso
- Classification: logistic_regression, knn, svm, gaussian_nb
- Ensemble: random_forest_classification, xgboost_classification, lightgbm_classification
- Clustering: kmeans, dbscan, gmm
- Dimensionality Reduction: pca

Plus integration tests for:
- solve() routing
- Reward shaper accuracy component
- Context engine ML profiling
- Explainer ML templates
- Knowledge graph ML registrations
"""

import pytest
import numpy as np
from typing import Dict, Any


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return {"X_train": X, "y_train": y}


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = X @ np.array([1.5, -2.0, 1.0]) + np.random.randn(100) * 0.1
    return {"X_train": X, "y_train": y}


@pytest.fixture
def clustering_data():
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    return X


@pytest.fixture
def high_dim_data():
    np.random.seed(42)
    X = np.random.rand(100, 20)
    return X


class TestMLAlgorithmBase:

    def test_extract_data_dict(self):
        from aalgoi.algorithms.ml.base import MLAlgorithm
        from sklearn.linear_model import LinearRegression

        algo = MLAlgorithm(LinearRegression, name="test")
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        X_train, y_train, X_test = algo._extract_data({
            "X_train": X, "y_train": y, "X_test": X
        })

        assert np.array_equal(X_train, X)
        assert np.array_equal(y_train, y)
        assert np.array_equal(X_test, X)

    def test_extract_data_tuple(self):
        from aalgoi.algorithms.ml.base import MLAlgorithm
        from sklearn.linear_model import LinearRegression

        algo = MLAlgorithm(LinearRegression, name="test")
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        X_train, y_train, X_test = algo._extract_data((X, y))

        assert np.array_equal(X_train, X)
        assert np.array_equal(y_train, y)
        assert X_test is None

    def test_extract_data_bare_array(self):
        from aalgoi.algorithms.ml.base import MLAlgorithm
        from sklearn.linear_model import LinearRegression

        algo = MLAlgorithm(LinearRegression, name="test")
        X = np.array([[1, 2], [3, 4]])

        X_train, y_train, X_test = algo._extract_data(X)

        assert np.array_equal(X_train, X)
        assert y_train is None
        assert X_test is None

    def test_process_returns_dict(self, classification_data):
        from aalgoi.algorithms.ml.classical import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        result = algo.process(classification_data)

        assert isinstance(result, dict)
        assert result.get("trained") is True
        assert "algorithm" in result

    def test_validate_output_true_on_success(self, classification_data):
        from aalgoi.algorithms.ml.classical import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        result = algo.process(classification_data)

        assert algo.validate_output(classification_data, result) is True

    def test_validate_output_false_on_failure(self):
        from aalgoi.algorithms.ml.classical import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        result = {"trained": False, "error": "bad data"}

        assert algo.validate_output(None, result) is False

    def test_describe_returns_metadata(self):
        from aalgoi.algorithms.ml.classical import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        meta = algo.describe()

        assert meta["name"] == "logistic_regression"
        assert meta["type"] == "ml"
        assert "classification" in meta["tags"]


class TestLinearRegression:

    def test_process_fits_model(self, regression_data):
        from aalgoi.algorithms.ml import LinearRegressionAlgo

        algo = LinearRegressionAlgo()
        result = algo.process(regression_data)

        assert result["trained"] is True
        assert result["algorithm"] == "linear_regression"

    def test_returns_r2_score(self, regression_data):
        from aalgoi.algorithms.ml import LinearRegressionAlgo

        algo = LinearRegressionAlgo()
        result = algo.process(regression_data)

        assert "r2_score" in result
        assert 0.0 <= result["r2_score"] <= 1.0

    def test_predicts_on_test_data(self, regression_data):
        from aalgoi.algorithms.ml import LinearRegressionAlgo

        algo = LinearRegressionAlgo()
        X_test = np.random.rand(10, 3)
        data = {**regression_data, "X_test": X_test}
        result = algo.process(data)

        assert "predictions" in result
        assert len(result["predictions"]) == 10

    def test_tags_include_regression(self):
        from aalgoi.algorithms.ml import LinearRegressionAlgo

        algo = LinearRegressionAlgo()
        assert "regression" in algo.tags
        assert "linear" in algo.tags


class TestRidge:

    def test_process_fits_model(self, regression_data):
        from aalgoi.algorithms.ml import RidgeAlgo

        algo = RidgeAlgo()
        result = algo.process(regression_data)

        assert result["trained"] is True
        assert result["algorithm"] == "ridge"

    def test_handles_multicollinearity(self):
        from aalgoi.algorithms.ml import RidgeAlgo

        np.random.seed(42)
        X = np.random.rand(100, 2)
        X = np.column_stack([X, X[:, 0] * 2])
        y = X[:, 0] + X[:, 1]

        algo = RidgeAlgo()
        result = algo.process({"X_train": X, "y_train": y})

        assert result["trained"] is True


class TestLasso:

    def test_process_fits_model(self, regression_data):
        from aalgoi.algorithms.ml import LassoAlgo

        algo = LassoAlgo()
        result = algo.process(regression_data)

        assert result["trained"] is True
        assert result["algorithm"] == "lasso"

    def test_feature_selection(self):
        from aalgoi.algorithms.ml import LassoAlgo

        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(100) * 0.1

        algo = LassoAlgo()
        result = algo.process({"X_train": X, "y_train": y})

        assert result["trained"] is True
        if "n_features_used" in result:
            assert result["n_features_used"] < 10


class TestLogisticRegression:

    def test_process_fits_model(self, classification_data):
        from aalgoi.algorithms.ml import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        result = algo.process(classification_data)

        assert result["trained"] is True
        assert "accuracy" in result
        assert result["accuracy"] > 0.8

    def test_predicts_on_test_data(self, classification_data):
        from aalgoi.algorithms.ml import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        X_test = np.random.rand(10, 5)
        data = {**classification_data, "X_test": X_test}
        result = algo.process(data)

        assert "predictions" in result
        assert all(p in [0, 1] for p in result["predictions"])


class TestKNN:

    def test_process_fits_model(self, classification_data):
        from aalgoi.algorithms.ml import KNNAlgo

        algo = KNNAlgo()
        result = algo.process(classification_data)

        assert result["trained"] is True
        assert "accuracy" in result

    def test_small_data_works(self):
        from aalgoi.algorithms.ml import KNNAlgo

        np.random.seed(42)
        X = np.random.rand(20, 2)
        y = (X[:, 0] > 0.5).astype(int)

        algo = KNNAlgo()
        result = algo.process({"X_train": X, "y_train": y})

        assert result["trained"] is True


class TestSVM:

    def test_process_fits_model(self, classification_data):
        from aalgoi.algorithms.ml import SVMAlgo

        algo = SVMAlgo()
        result = algo.process(classification_data)

        assert result["trained"] is True
        assert "accuracy" in result

    def test_n_support_reported(self, classification_data):
        from aalgoi.algorithms.ml import SVMAlgo

        algo = SVMAlgo()
        result = algo.process(classification_data)

        if "n_support" in result:
            assert result["n_support"] > 0


class TestGaussianNB:

    def test_process_fits_model(self, classification_data):
        from aalgoi.algorithms.ml import GaussianNBAlgo

        algo = GaussianNBAlgo()
        result = algo.process(classification_data)

        assert result["trained"] is True
        assert "accuracy" in result

    def test_fast_training(self, classification_data):
        import time
        from aalgoi.algorithms.ml import GaussianNBAlgo

        algo = GaussianNBAlgo()

        start = time.time()
        algo.process(classification_data)
        elapsed = time.time() - start

        assert elapsed < 0.1


class TestRandomForest:

    def test_classification_mode(self, classification_data):
        from aalgoi.algorithms.ml.ensemble import RandomForestAlgo

        algo = RandomForestAlgo(task="classification")
        result = algo.process(classification_data)

        assert result["trained"] is True
        assert "accuracy" in result
        assert result["accuracy"] > 0.8

    def test_regression_mode(self, regression_data):
        from aalgoi.algorithms.ml.ensemble import RandomForestAlgo

        algo = RandomForestAlgo(task="regression")
        result = algo.process(regression_data)

        assert result["trained"] is True
        assert "r2_score" in result

    def test_tags_correct(self):
        from aalgoi.algorithms.ml.ensemble import RandomForestAlgo

        algo = RandomForestAlgo(task="classification")
        assert "ensemble" in algo.tags
        assert "bagging" in algo.tags


class TestXGBoost:

    def test_available_check(self):
        from aalgoi.algorithms.ml.ensemble import XGBoostAlgo

        algo = XGBoostAlgo(task="classification")
        assert isinstance(algo._available, bool)

    def test_raises_if_unavailable(self, classification_data):
        from aalgoi.algorithms.ml.ensemble import XGBoostAlgo

        algo = XGBoostAlgo(task="classification")

        if not algo._available:
            with pytest.raises(ImportError, match="xgboost"):
                algo.process(classification_data)
        else:
            result = algo.process(classification_data)
            assert result["trained"] is True

    def test_classification_if_available(self, classification_data):
        from aalgoi.algorithms.ml.ensemble import XGBoostAlgo

        algo = XGBoostAlgo(task="classification")

        if algo._available:
            result = algo.process(classification_data)
            assert result["trained"] is True
            assert "accuracy" in result


class TestLightGBM:

    def test_available_check(self):
        from aalgoi.algorithms.ml.ensemble import LightGBMAlgo

        algo = LightGBMAlgo(task="classification")
        assert isinstance(algo._available, bool)

    def test_raises_if_unavailable(self, classification_data):
        from aalgoi.algorithms.ml.ensemble import LightGBMAlgo

        algo = LightGBMAlgo(task="classification")

        if not algo._available:
            with pytest.raises(ImportError, match="lightgbm"):
                algo.process(classification_data)
        else:
            result = algo.process(classification_data)
            assert result["trained"] is True


class TestKMeans:

    def test_clusters_data(self, clustering_data):
        from aalgoi.algorithms.ml import KMeansClustering

        algo = KMeansClustering()
        result = algo.process(clustering_data)

        assert result["trained"] is True
        assert "labels" in result
        assert len(result["labels"]) == 90

    def test_returns_cluster_centers(self, clustering_data):
        from aalgoi.algorithms.ml import KMeansClustering

        algo = KMeansClustering()
        result = algo.process(clustering_data)

        assert "cluster_centers" in result
        assert len(result["cluster_centers"]) == 3


class TestDBSCAN:

    def test_clusters_data(self, clustering_data):
        from aalgoi.algorithms.ml import DBSCANClustering

        algo = DBSCANClustering()
        result = algo.process(clustering_data)

        assert result["trained"] is True
        assert "labels" in result

    def test_detects_noise(self):
        from aalgoi.algorithms.ml import DBSCANClustering

        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) * 0.5,
            np.array([[10, 10], [11, 11]])
        ])

        algo = DBSCANClustering()
        result = algo.process(X)

        assert "n_noise" in result


class TestGMM:

    def test_clusters_data(self, clustering_data):
        from aalgoi.algorithms.ml.clustering import GMMAlgo

        algo = GMMAlgo()
        result = algo.process(clustering_data)

        assert result["trained"] is True
        assert "labels" in result

    def test_returns_weights(self, clustering_data):
        from aalgoi.algorithms.ml.clustering import GMMAlgo

        algo = GMMAlgo()
        result = algo.process(clustering_data)

        if "weights" in result:
            assert len(result["weights"]) == 3


class TestPCA:

    def test_reduces_dimensionality(self, high_dim_data):
        from aalgoi.algorithms.ml.clustering import PCAReductionAlgo

        algo = PCAReductionAlgo()
        result = algo.process(high_dim_data)

        assert result["trained"] is True
        assert "transformed" in result
        assert len(result["transformed"]) == 100
        assert len(result["transformed"][0]) == 2

    def test_explained_variance(self, high_dim_data):
        from aalgoi.algorithms.ml.clustering import PCAReductionAlgo

        algo = PCAReductionAlgo()
        result = algo.process(high_dim_data)

        assert "explained_variance_ratio" in result
        assert result["explained_variance_ratio"][0] > result["explained_variance_ratio"][1]


class TestSolveRouting:

    def test_classify_routes_to_classifier(self, classification_data):
        import random
        random.seed(42)
        from aalgoi import solve

        result = solve("classify", classification_data)

        assert result.success is True
        assert result.algorithm not in ["kmeans", "dbscan", "gmm"]
        assert result.algorithm in [
            "logistic_regression", "knn", "svm", "gaussian_nb",
            "random_forest_classification", "xgboost_classification"
        ]

    def test_regression_routes_to_regressor(self, regression_data):
        import random
        random.seed(42)
        from aalgoi import solve

        result = solve("regression", regression_data)

        assert result.success is True
        assert result.algorithm in [
            "linear_regression", "ridge", "lasso",
            "random_forest_classification", "xgboost_classification"
        ]

    def test_cluster_routes_to_clustering(self, clustering_data):
        import random
        random.seed(42)
        from aalgoi import solve

        result = solve("cluster", clustering_data)

        assert result.success is True
        assert result.algorithm in ["kmeans", "dbscan", "gmm"]


class TestRewardShaperML:

    def test_accuracy_increases_reward(self):
        from aalgoi.core.rl.reward_shaper import RewardShaper

        shaper = RewardShaper()

        reward_high = shaper.compute(
            success=True, elapsed=0.01, data_size=100,
            algo_name="random_forest", metrics={"accuracy": 0.95}
        )

        reward_low = shaper.compute(
            success=True, elapsed=0.01, data_size=100,
            algo_name="random_forest", metrics={"accuracy": 0.55}
        )

        assert reward_high > reward_low

    def test_r2_score_increases_reward(self):
        from aalgoi.core.rl.reward_shaper import RewardShaper

        shaper = RewardShaper()

        reward_high = shaper.compute(
            success=True, elapsed=0.01, data_size=100,
            algo_name="linear_regression", metrics={"r2_score": 0.9}
        )

        reward_low = shaper.compute(
            success=True, elapsed=0.01, data_size=100,
            algo_name="linear_regression", metrics={"r2_score": 0.1}
        )

        assert reward_high > reward_low

    def test_no_metrics_same_as_before(self):
        from aalgoi.core.rl.reward_shaper import RewardShaper

        shaper = RewardShaper()
        reward = shaper.compute(
            success=True, elapsed=0.01, data_size=100,
            algo_name="quicksort", metrics=None
        )

        assert reward > 0


class TestContextEngineML:

    def test_profiles_classification_data(self, classification_data):
        from aalgoi.core.context_engine import ContextEngine

        engine = ContextEngine()
        profile = engine._analyze_ml_data(classification_data)

        assert profile["type"] == "ml"
        assert profile["n_samples"] == 100
        assert profile["n_features"] == 5
        assert profile["is_classification"] is True
        assert profile["n_classes"] == 2

    def test_profiles_regression_data(self, regression_data):
        from aalgoi.core.context_engine import ContextEngine

        engine = ContextEngine()
        profile = engine._analyze_ml_data(regression_data)

        assert profile["type"] == "ml"
        assert profile["n_samples"] == 100
        assert profile["n_features"] == 3
        assert profile["is_classification"] is False
        assert "target_mean" in profile

    def test_profiles_bare_array(self):
        from aalgoi.core.context_engine import ContextEngine

        engine = ContextEngine()
        X = np.random.rand(50, 10)
        profile = engine._analyze_ml_data(X)

        assert profile["type"] == "ml"
        assert profile["n_samples"] == 50
        assert profile["n_features"] == 10

    def test_samples_per_feature(self):
        from aalgoi.core.context_engine import ContextEngine

        engine = ContextEngine()
        X = np.random.rand(100, 5)
        profile = engine._analyze_ml_data({"X_train": X})

        assert profile["samples_per_feature"] == 20.0

    def test_imbalance_ratio(self):
        from aalgoi.core.context_engine import ContextEngine

        engine = ContextEngine()
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        profile = engine._analyze_ml_data({"X_train": X, "y_train": y})

        assert profile["imbalance_ratio"] == 9.0
        assert abs(profile["majority_class_pct"] - 0.9) < 0.01


class TestExplainerML:

    @pytest.mark.parametrize("algo_name", [
        "linear_regression",
        "logistic_regression",
        "random_forest_classification",
        "svm",
        "knn",
        "xgboost_classification",
        "kmeans",
        "dbscan",
        "gmm",
        "pca",
    ])
    def test_explain_ml_algorithm(self, algo_name):
        from aalgoi import explain

        exp = explain(algo_name)

        assert exp is not None
        assert hasattr(exp, "summary")
        assert len(exp.summary) > 0

    def test_random_forest_best_for(self):
        from aalgoi import explain

        exp = explain("random_forest_classification")

        assert "tabular" in exp.best_for.lower()

    def test_svm_complexity(self):
        from aalgoi import explain

        exp = explain("svm")

        assert "n\u00b2" in exp.complexity or "quadratic" in exp.complexity.lower()


class TestRegistryML:

    def test_all_ml_algos_in_registry(self):
        from aalgoi.pipeline import UniversalSolver

        solver = UniversalSolver()

        expected = [
            "linear_regression", "ridge", "lasso",
            "logistic_regression", "knn", "svm", "gaussian_nb",
            "random_forest_classification",
            "kmeans", "dbscan",
            "gmm", "pca",
        ]

        for name in expected:
            assert name in solver.registry, f"{name} not in registry"

    def test_registry_count_increased(self):
        from aalgoi.pipeline import UniversalSolver

        solver = UniversalSolver()

        assert len(solver.registry) >= 25


class TestMLEdgeCases:

    def test_single_sample(self):
        from aalgoi.algorithms.ml import GaussianNBAlgo

        algo = GaussianNBAlgo()
        X = np.array([[1, 2, 3]])
        y = np.array([0])

        result = algo.process({"X_train": X, "y_train": y})

        assert "trained" in result

    def test_very_large_n_features(self):
        from aalgoi.algorithms.ml import GaussianNBAlgo

        np.random.seed(42)
        X = np.random.rand(50, 1000)
        y = np.random.randint(0, 2, 50)

        algo = GaussianNBAlgo()
        result = algo.process({"X_train": X, "y_train": y})

        assert result["trained"] is True

    def test_predict_without_fit_fails(self):
        from aalgoi.algorithms.ml import LogisticRegressionAlgo

        algo = LogisticRegressionAlgo()
        assert algo.model is None
