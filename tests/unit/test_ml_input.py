import pytest
import numpy as np
from aalgoi.algorithms.ml.base import MLAlgorithm


class DummyModel:
    def __init__(self, **kwargs):
        self.fitted = False

    def fit(self, X, y=None):
        self.fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


class DummyMLAlgo(MLAlgorithm):
    def __init__(self):
        super().__init__(model_class=DummyModel, name="dummy_ml")


def test_1d_input_reshaped_to_2d():
    algo = DummyMLAlgo()
    X_train, y_train, X_test = algo._extract_data([1, 2, 3, 4, 5])
    assert X_train is not None
    assert X_train.ndim == 2, f"Expected 2D, got {X_train.ndim}D"
    assert X_train.shape[1] == 1


def test_2d_input_preserved():
    algo = DummyMLAlgo()
    X = np.random.randn(10, 3)
    X_train, y_train, X_test = algo._extract_data(X)
    assert X_train is not None
    assert X_train.ndim == 2
    assert X_train.shape == (10, 3)


def test_dict_input_preserves_shape():
    algo = DummyMLAlgo()
    X = np.random.randn(10, 5)
    data = {"X_train": X, "y_train": np.random.randn(10)}
    X_train, y_train, X_test = algo._extract_data(data)
    assert X_train is not None
    assert X_train.ndim == 2
    assert X_train.shape == (10, 5)


def test_1d_in_dict_reshaped():
    algo = DummyMLAlgo()
    data = {"X_train": [1, 2, 3, 4, 5]}
    X_train, y_train, X_test = algo._extract_data(data)
    assert X_train.ndim == 2


def test_1d_test_also_reshaped():
    algo = DummyMLAlgo()
    data = {"X_train": np.random.randn(10, 3), "X_test": [1, 2, 3]}
    X_train, y_train, X_test = algo._extract_data(data)
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert X_test.shape[1] == 1
