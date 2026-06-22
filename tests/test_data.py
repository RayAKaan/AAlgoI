"""Tests for aalgoi._data module."""
import pytest
from aalgoi._data import normalize, detect_type, normalize_with_metadata
import numpy as np


class TestNormalize:
    def test_none(self):
        assert normalize(None) is None

    def test_primitives(self):
        assert normalize(42) == 42
        assert normalize(3.14) == 3.14
        assert normalize("hello") == "hello"
        assert normalize(True) is True
        assert normalize(False) is False

    def test_list(self):
        assert normalize([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert normalize((1, 2, 3)) == [1, 2, 3]

    def test_set(self):
        result = normalize({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_dict(self):
        result = normalize({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested(self):
        result = normalize({"a": [1, 2, {"b": 3}]})
        assert result == {"a": [1, 2, {"b": 3}]}

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = normalize(arr)
        assert result == [1, 2, 3]

    def test_idempotent(self):
        data = {"a": [1, 2, 3]}
        assert normalize(normalize(data)) == normalize(data)


class TestDetectType:
    def test_none(self):
        assert detect_type(None) == "none"

    def test_int(self):
        assert detect_type(42) == "int"

    def test_list(self):
        assert detect_type([1, 2, 3]) == "list(3)"

    def test_dict(self):
        assert detect_type({"a": 1}) == "dict(1 keys)"


class TestNormalizeWithMetadata:
    def test_basic(self):
        result = normalize_with_metadata([1, 2, 3])
        assert result["data"] == [1, 2, 3]
        assert result["original_type"] == "list(3)"

    def test_numpy(self):
        result = normalize_with_metadata(np.array([1, 2, 3]))
        assert result["data"] == [1, 2, 3]
        assert "numpy" in result["original_type"].lower()
