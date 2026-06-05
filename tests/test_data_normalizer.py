"""Tests for aalgoi._data universal data normalizer."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from fractions import Fraction

import pytest

from aalgoi._data import detect_type, normalize, normalize_with_metadata


class Color(Enum):
    RED = 1
    GREEN = 2


@dataclass
class Point:
    x: float
    y: float


# ── Primitives ───────────────────────────────────────────────────────

class TestPrimitives:
    def test_none(self):
        assert normalize(None) is None

    def test_bool(self):
        assert normalize(True) is True
        assert normalize(False) is False

    def test_int(self):
        assert normalize(42) == 42

    def test_float(self):
        assert normalize(3.14) == 3.14

    def test_str(self):
        assert normalize("hello") == "hello"


# ── Collections ──────────────────────────────────────────────────────

class TestCollections:
    def test_list(self):
        assert normalize([3, 1, 2]) == [3, 1, 2]

    def test_nested_list(self):
        assert normalize([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_tuple(self):
        assert normalize((1, 2, 3)) == [1, 2, 3]

    def test_set(self):
        result = normalize({3, 1, 2})
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_frozenset(self):
        result = normalize(frozenset([3, 1, 2]))
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_dict(self):
        result = normalize({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = normalize({"a": {"b": [1, 2]}})
        assert result == {"a": {"b": [1, 2]}}


# ── Standard library types ───────────────────────────────────────────

class TestStdlibTypes:
    def test_datetime(self):
        d = datetime(2024, 1, 15, 10, 30, 0)
        assert normalize(d) == "2024-01-15T10:30:00"

    def test_date(self):
        d = date(2024, 1, 15)
        assert normalize(d) == "2024-01-15"

    def test_timedelta(self):
        assert normalize(timedelta(hours=2)) == 7200.0

    def test_decimal(self):
        assert normalize(Decimal("3.14")) == 3.14

    def test_fraction(self):
        assert normalize(Fraction(1, 3)) == pytest.approx(0.333333, abs=1e-5)

    def test_enum(self):
        assert normalize(Color.RED) == 1

    def test_range(self):
        result = normalize(range(5))
        assert result == {"type": "range", "start": 0, "stop": 5, "step": 1}

    def test_bytes_json(self):
        assert normalize(b'{"key": "value"}') == {"key": "value"}

    def test_bytes_csv(self):
        data = b"a,b\n1,2\n3,4"
        result = normalize(data)
        assert result["columns"] == ["a", "b"]
        assert result["shape"] == [2, 2]


# ── Dataclass / Pydantic / Namedtuple ────────────────────────────────

class TestStructured:
    def test_dataclass(self):
        p = Point(3.0, 4.0)
        assert normalize(p) == {"x": 3.0, "y": 4.0}

    def test_namedtuple(self):
        from collections import namedtuple
        N = namedtuple("N", ["a", "b"])
        result = normalize(N(1, 2))
        # namedtuple is a tuple subclass, normalizes as list
        assert result == [1, 2]

    def test_pydantic(self):
        try:
            from pydantic import BaseModel

            class User(BaseModel):
                name: str
                age: int

            result = normalize(User(name="Alice", age=30))
            assert result == {"name": "Alice", "age": 30}
        except ImportError:
            pytest.skip("pydantic not installed")


# ── Generator / Iterator ─────────────────────────────────────────────

class TestGenerators:
    def test_generator(self):
        def gen():
            yield 1
            yield 2
            yield 3
        assert normalize(gen()) == [1, 2, 3]

    def test_generator_capped(self):
        def big():
            yield from range(20_000)
        result = normalize(big())
        assert len(result) == 10_000


# ── detect_type ──────────────────────────────────────────────────────

class TestDetectType:
    def test_none(self):
        assert detect_type(None) == "none"

    def test_bool(self):
        assert detect_type(True) == "bool"

    def test_int(self):
        assert detect_type(42) == "int"

    def test_float(self):
        assert detect_type(3.14) == "float"

    def test_str(self):
        assert detect_type("hello") == "str"

    def test_list(self):
        assert detect_type([1, 2]) == "list(2)"

    def test_dict(self):
        assert detect_type({"a": 1}) == "dict(1 keys)"

    def test_set(self):
        assert detect_type({1, 2}) == "set(2)"


# ── normalize_with_metadata ──────────────────────────────────────────

class TestNormalizeWithMetadata:
    def test_list(self):
        result = normalize_with_metadata([1, 2, 3])
        assert result["original_type"] == "list(3)"
        assert result["data"] == [1, 2, 3]

    def test_dict(self):
        result = normalize_with_metadata({"a": 1})
        assert result["original_type"] == "dict(1 keys)"
        assert result["data"] == {"a": 1}
