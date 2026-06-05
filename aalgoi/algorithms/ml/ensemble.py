
from aalgoi.algorithms.ml.base import MLAlgorithm
import logging

logger = logging.getLogger(__name__)


class RandomForestAlgo(MLAlgorithm):
    """Random Forest classifier or regressor."""

    def __init__(self, task: str = "classification"):
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier as Model
            params = {"n_estimators": 100, "max_depth": 10, "n_jobs": -1, "random_state": 42}
        else:
            from sklearn.ensemble import RandomForestRegressor as Model
            params = {"n_estimators": 100, "max_depth": 10, "n_jobs": -1, "random_state": 42}

        super().__init__(
            Model,
            params,
            name="random_forest_{}".format(task),
            task=task
        )
        self.tags = ["ml", "ensemble", "bagging", "decision tree", task]
        self.best_for = ["tabular", "nonlinear", "mixed_features", "feature_importance"]
        self.time_complexity = "O(n*m*log(n))"
        self.space_complexity = "O(n*m)"
        self.patterns = ["Ensemble", "TreeBased", "Bagging", "DecisionTree"]
        self.problem_types = ["CLASSIFICATION"] if task == "classification" else ["REGRESSION"]

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            if self.task == "classification":
                from sklearn.metrics import accuracy_score
                preds = self.model.predict(X)
                acc = float(accuracy_score(y, preds))
                return {"accuracy": acc}
            else:
                r2 = float(self.model.score(X, y))
                return {"r2_score": r2}
        except Exception:
            return {}


class XGBoostAlgo(MLAlgorithm):
    """XGBoost gradient boosting classifier or regressor."""

    _available_cache = {}

    def __init__(self, task: str = "classification"):
        if task not in self._available_cache:
            try:
                if task == "classification":
                    from xgboost import XGBClassifier
                else:
                    from xgboost import XGBRegressor
                self._available_cache[task] = True
            except ImportError:
                logger.info("xgboost not installed \u2014 XGBoostAlgo(%s) unavailable", task)
                self._available_cache[task] = False

        self._available = self._available_cache.get(task, False)

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
        }

        Model = None
        if self._available:
            try:
                if task == "classification":
                    from xgboost import XGBClassifier as Model
                    params["use_label_encoder"] = False
                    params["eval_metric"] = "logloss"
                else:
                    from xgboost import XGBRegressor as Model
            except ImportError:
                self._available = False

        super().__init__(
            Model,
            params,
            name="xgboost_{}".format(task),
            task=task
        )

        self.tags = ["ml", "ensemble", "boosting", "gradient_boosting", task]
        self.best_for = ["tabular", "competitions", "accuracy_critical", "feature_importance"]
        self.time_complexity = "O(n*m*log(n))"
        self.space_complexity = "O(n*m)"
        self.patterns = ["Ensemble", "Boosting", "GradientBoosting"]
        self.problem_types = ["CLASSIFICATION"] if task == "classification" else ["REGRESSION"]

    def process(self, data):
        if not self._available:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost"
            )
        return super().process(data)

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            if self.task == "classification":
                from sklearn.metrics import accuracy_score
                preds = self.model.predict(X)
                acc = float(accuracy_score(y, preds))
                return {"accuracy": acc}
            else:
                r2 = float(self.model.score(X, y))
                return {"r2_score": r2}
        except Exception:
            return {}


class LightGBMAlgo(MLAlgorithm):
    """LightGBM gradient boosting classifier or regressor."""

    _available_cache = {}

    def __init__(self, task: str = "classification"):
        if task not in self._available_cache:
            try:
                if task == "classification":
                    from lightgbm import LGBMClassifier
                else:
                    from lightgbm import LGBMRegressor
                self._available_cache[task] = True
            except ImportError:
                logger.info("lightgbm not installed \u2014 LightGBMAlgo(%s) unavailable", task)
                self._available_cache[task] = False

        self._available = self._available_cache.get(task, False)

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }

        Model = None
        if self._available:
            try:
                if task == "classification":
                    from lightgbm import LGBMClassifier as Model
                else:
                    from lightgbm import LGBMRegressor as Model
            except ImportError:
                self._available = False

        super().__init__(
            Model,
            params,
            name="lightgbm_{}".format(task),
            task=task
        )

        self.tags = ["ml", "ensemble", "boosting", "gradient_boosting", task, "fast"]
        self.best_for = ["large_data", "tabular", "speed_critical"]
        self.time_complexity = "O(n*m*log(n))"
        self.space_complexity = "O(n*m)"
        self.patterns = ["Ensemble", "Boosting", "GradientBoosting"]
        self.problem_types = ["CLASSIFICATION"] if task == "classification" else ["REGRESSION"]

    def process(self, data):
        if not self._available:
            raise ImportError(
                "LightGBM not installed. Run: pip install lightgbm"
            )
        return super().process(data)

    def _compute_metrics(self, X, y, X_test):
        if self.model is None or y is None:
            return {}
        try:
            if self.task == "classification":
                from sklearn.metrics import accuracy_score
                preds = self.model.predict(X)
                acc = float(accuracy_score(y, preds))
                return {"accuracy": acc}
            else:
                r2 = float(self.model.score(X, y))
                return {"r2_score": r2}
        except Exception:
            return {}
