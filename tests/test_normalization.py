"""Comprehensive tests for aalgoi._data normalization."""
import pytest
import csv
import json
from datetime import datetime, date, time as dt_time, timedelta
from decimal import Decimal
from fractions import Fraction
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass

import numpy as np

from aalgoi._data import normalize, detect_type, normalize_with_metadata


class TestNormalizePrimitives:
    def test_none(self):
        assert normalize(None) is None

    def test_bool(self):
        assert normalize(True) is True
        assert normalize(False) is False

    def test_int(self):
        assert normalize(42) == 42
        assert normalize(-100) == -100
        assert normalize(0) == 0

    def test_float(self):
        assert normalize(3.14) == 3.14
        assert normalize(-0.0) == 0.0
        assert normalize(float('inf')) == float('inf')
        assert normalize(float('nan')) != normalize(float('nan'))

    def test_string(self):
        assert normalize("hello") == "hello"
        assert normalize("") == ""
        assert normalize("  spaced  ") == "  spaced  "


class TestNormalizeContainers:
    def test_list(self):
        assert normalize([1, 2, 3]) == [1, 2, 3]
        assert normalize([]) == []
        assert normalize([1, "two", 3.0]) == [1, "two", 3.0]

    def test_tuple(self):
        assert normalize((1, 2, 3)) == [1, 2, 3]
        assert normalize(()) == []

    def test_dict(self):
        assert normalize({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        assert normalize({}) == {}

    def test_set(self):
        result = normalize({3, 1, 2})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_frozenset(self):
        result = normalize(frozenset({3, 1, 2}))
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_ordered_dict(self):
        od = OrderedDict([("a", 1), ("b", 2)])
        result = normalize(od)
        assert isinstance(result, dict)
        assert result == {"a": 1, "b": 2}

    def test_nested_containers(self):
        data = {"a": [1, 2, {"b": (3, 4)}], "c": {5, 6}}
        result = normalize(data)
        assert result["a"] == [1, 2, {"b": [3, 4]}]
        assert isinstance(result["c"], list)

    def test_deeply_nested(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = normalize(data)
        assert result == data


class TestNormalizeStdlibTypes:
    def test_decimal(self):
        assert normalize(Decimal("3.14159")) == pytest.approx(3.14159)

    def test_fraction(self):
        assert normalize(Fraction(1, 2)) == pytest.approx(0.5)

    def test_complex(self):
        result = normalize(complex(1, 2))
        assert result == {"real": 1.0, "imag": 2.0}

    def test_range(self):
        result = normalize(range(1, 10, 2))
        assert result == {"start": 1, "stop": 10, "step": 2, "type": "range"}

    def test_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = normalize(dt)
        assert result == "2024-01-15T10:30:00"

    def test_date(self):
        d = date(2024, 1, 15)
        result = normalize(d)
        assert result == "2024-01-15"

    def test_time(self):
        t = dt_time(10, 30, 0)
        result = normalize(t)
        assert result == "10:30:00"

    def test_timedelta(self):
        td = timedelta(hours=1, minutes=30)
        result = normalize(td)
        assert result == pytest.approx(5400.0)


class TestNormalizeEnum:
    def test_enum_value(self):
        class Color(Enum):
            RED = 1
            GREEN = 2
        assert normalize(Color.RED) == 1
        assert normalize(Color.GREEN) == 2

    def test_string_enum(self):
        class Status(Enum):
            OK = "ok"
            ERROR = "error"
        assert normalize(Status.OK) == "ok"


class TestNormalizeDataclass:
    def test_dataclass(self):
        @dataclass
        class Person:
            name: str
            age: int
        p = Person("Alice", 30)
        result = normalize(p)
        assert result == {"name": "Alice", "age": 30}

    def test_nested_dataclass(self):
        @dataclass
        class Address:
            city: str
        @dataclass
        class Person:
            name: str
            addr: Address
        p = Person("Bob", Address("NYC"))
        result = normalize(p)
        assert result == {"name": "Bob", "addr": {"city": "NYC"}}


class TestNormalizeNumpy:
    def test_0d_array(self):
        arr = np.array(42)
        assert normalize(arr) == 42

    def test_1d_array(self):
        arr = np.array([1, 2, 3])
        assert normalize(arr) == [1, 2, 3]

    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        assert normalize(arr) == [[1, 2], [3, 4]]

    def test_float_array(self):
        arr = np.array([1.5, 2.5, 3.5])
        result = normalize(arr)
        assert result == pytest.approx([1.5, 2.5, 3.5])


class TestNormalizeString:
    def test_json_string(self):
        result = normalize('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_array_string(self):
        result = normalize('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_nested_json_string(self):
        result = normalize('{"a": [1, 2, {"b": 3}]}')
        assert result == {"a": [1, 2, {"b": 3}]}

    def test_invalid_json_string(self):
        result = normalize('{invalid json}')
        assert result == '{invalid json}'

    def test_plain_string(self):
        result = normalize("hello world")
        assert result == "hello world"


class TestNormalizeFileSecurity:
    def test_file_path_not_read_by_default(self):
        result = normalize("/etc/passwd")
        assert result == "/etc/passwd"

    def test_home_path_not_read_by_default(self):
        result = normalize("~/.ssh/id_rsa")
        assert result == "~/.ssh/id_rsa"

    def test_relative_path_not_read_by_default(self):
        result = normalize("./config.json")
        assert result == "./config.json"

    def test_file_read_when_explicitly_allowed(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')
        result = normalize(str(test_file), allow_file_read=True)
        assert result == {"key": "value"}

    def test_csv_file_read_when_allowed(self, tmp_path):
        test_file = tmp_path / "test.csv"
        with open(test_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age'])
            writer.writerow(['Alice', 30])
            writer.writerow(['Bob', 25])
        result = normalize(str(test_file), allow_file_read=True)
        assert "columns" in result
        assert result["columns"] == ["name", "age"]
        assert len(result["rows"]) == 2

    def test_txt_file_read_when_allowed(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = normalize(str(test_file), allow_file_read=True)
        assert result == "hello world"

    def test_nonexistent_file_returns_path(self):
        result = normalize("/nonexistent/file.json", allow_file_read=True)
        assert result == "/nonexistent/file.json"


class TestNormalizeURLSecurity:
    def test_url_not_fetched_by_default(self):
        result = normalize("https://example.com/data.json")
        assert result == "https://example.com/data.json"

    def test_http_url_not_fetched_by_default(self):
        result = normalize("http://example.com/data.json")
        assert result == "http://example.com/data.json"


class TestNormalizeIdempotence:
    def test_idempotent_primitives(self):
        for val in [42, 3.14, "hello", True, False, None]:
            assert normalize(normalize(val)) == normalize(val), f"Failed for {val}"

    def test_idempotent_list(self):
        data = [1, 2, 3]
        assert normalize(normalize(data)) == normalize(data)

    def test_idempotent_dict(self):
        data = {"a": 1, "b": [2, 3]}
        assert normalize(normalize(data)) == normalize(data)

    def test_idempotent_nested(self):
        data = {"a": [1, 2, {"b": 3}]}
        assert normalize(normalize(data)) == normalize(data)

    def test_idempotent_numpy(self):
        data = np.array([1, 2, 3])
        assert normalize(normalize(data)) == normalize(data)


class TestNormalizeBytes:
    def test_json_bytes(self):
        data = b'{"key": "value"}'
        result = normalize(data)
        assert result == {"key": "value"}

    def test_csv_bytes(self):
        data = b"name,age\nAlice,30\nBob,25"
        result = normalize(data)
        assert "columns" in result
        assert result["columns"] == ["name", "age"]

    def test_arbitrary_bytes(self):
        data = b'\x00\x01\x02\x03'
        result = normalize(data)
        assert isinstance(result, str)


class TestNormalizeGenerator:
    def test_generator(self):
        def gen():
            yield 1
            yield 2
            yield 3
        result = normalize(gen())
        assert result == [1, 2, 3]

    def test_generator_cap(self):
        def big_gen():
            for i in range(20000):
                yield i
        result = normalize(big_gen())
        assert len(result) == 10000
        assert result[0] == 0
        assert result[-1] == 9999


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
        assert detect_type([1, 2, 3]) == "list(3)"

    def test_empty_list(self):
        assert detect_type([]) == "list(0)"

    def test_dict(self):
        assert detect_type({"a": 1, "b": 2}) == "dict(2 keys)"

    def test_set(self):
        assert detect_type({1, 2, 3}) == "set(3)"

    def test_frozenset(self):
        assert detect_type(frozenset({1, 2})) == "frozenset(2)"

    def test_numpy_array(self):
        result = detect_type(np.array([1, 2, 3]))
        assert "numpy" in result.lower()

    def test_custom_object(self):
        class Foo:
            pass
        result = detect_type(Foo())
        assert "Foo" in result


class TestNormalizeWithMetadata:
    def test_list(self):
        result = normalize_with_metadata([1, 2, 3])
        assert result["data"] == [1, 2, 3]
        assert result["original_type"] == "list(3)"

    def test_dict(self):
        result = normalize_with_metadata({"a": 1})
        assert result["data"] == {"a": 1}
        assert result["original_type"] == "dict(1 keys)"

    def test_numpy(self):
        result = normalize_with_metadata(np.array([1, 2, 3]))
        assert result["data"] == [1, 2, 3]
        assert "numpy" in result["original_type"].lower()

    def test_string(self):
        result = normalize_with_metadata("hello")
        assert result["data"] == "hello"
        assert result["original_type"] == "str"
