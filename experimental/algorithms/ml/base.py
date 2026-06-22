
import logging
from typing import Any

import numpy as np

from aalgoi.algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class MLAlgorithm(Algorithm):
    """
    Base class for ML algorithms.

    Input data format for process():
        {
            "X_train": np.ndarray,
            "y_train": np.ndarray (optional for unsupervised),
            "X_test": np.ndarray (optional)
        }

    Returns:
        If X_test provided: dict with "predictions" key
        If no X_test: dict with training metrics
    """

    def __init__(
        self,
        model_class: type | None,
        default_params: dict | None = None,
        name: str | None = None,
        task: str = "auto"
    ) -> None:
        super().__init__()
        self.model_class = model_class
        self.default_params = default_params or {}
        self._name = name or (model_class.__name__ if model_class else "unknown")
        self.task = task
        self.model = None

        self.time_complexity = "O(n*d)"
        self.space_complexity = "O(d\u00b2)"
        self.tags: list[str] = ["ml", "supervised"]
        self.best_for: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def process(self, data: Any) -> Any:
        X_train, y_train, X_test = self._extract_data(data)

        if X_train is None:
            return {"trained": False, "error": "invalid data format"}

        try:
            self.model = self.model_class(**self.default_params)
        except Exception as e:
            logger.warning("%s init failed: %s", self._name, e)
            return {"trained": False, "error": str(e)}

        try:
            if y_train is not None:
                self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train)
        except Exception as e:
            logger.warning("%s fit failed: %s", self._name, e)
            return {"trained": False, "error": str(e)}

        result = {
            "trained": True,
            "algorithm": self._name,
        }

        if X_test is not None:
            try:
                result["predictions"] = self.model.predict(X_test).tolist()
            except Exception as e:
                logger.warning("%s predict failed: %s", self._name, e)
                result["predict_error"] = str(e)

        result.update(self._compute_metrics(X_train, y_train, X_test))

        return result

    @staticmethod
    def _validate_ml_input(X: Any) -> np.ndarray | None:
        if X is None:
            return None
        X = np.asarray(X)
        if X.ndim == 0:
            X = X.reshape(1, -1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _extract_data(self, data: Any) -> tuple:
        X_train = None
        y_train = None
        X_test = None

        if isinstance(data, dict):
            X_train = data.get("X_train")
            if X_train is None:
                X_train = data.get("data")
            y_train = data.get("y_train")
            X_test = data.get("X_test")

        elif isinstance(data, (list, tuple)):
            if len(data) >= 2:
                X_train, y_train = data[0], data[1]
                if len(data) >= 3:
                    X_test = data[2]
            elif len(data) == 1:
                X_train = data[0]

        elif hasattr(data, 'shape'):
            X_train = data

        X_train = self._validate_ml_input(X_train)
        X_test = self._validate_ml_input(X_test)
        return X_train, y_train, X_test

    def _compute_metrics(self, X: Any, y: Any, X_test: Any) -> dict:
        return {"status": "fitted"}

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if isinstance(output_data, dict):
            return output_data.get("trained", False) is True
        return False

    def describe(self) -> dict:
        return {
            "name": self._name,
            "type": "ml",
            "task": self.task,
            "tags": self.tags,
            "best_for": self.best_for,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "handles_sparse": True,
            "supports_online": False,
        }
