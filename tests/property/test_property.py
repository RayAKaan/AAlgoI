"""Property-based tests using Hypothesis."""
import pytest
from hypothesis import given, strategies as st, settings, assume
import tempfile
import warnings

from hypothesis import HealthCheck

from aalgoi._core import Mind
from aalgoi._result import SolveResult
from aalgoi._data import normalize, detect_type
from aalgoi import shortcuts

settings.register_profile("default", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("default")


class TestSolveResultProperties:
    @given(st.lists(st.integers(), max_size=20))
    def test_list_output_iteration_equals_list(self, data):
        r = SolveResult(output=data)
        assert list(r) == data

    @given(st.lists(st.integers(), max_size=20))
    def test_list_output_len_equals_len(self, data):
        r = SolveResult(output=data)
        assert len(r) == len(data)

    @given(st.integers())
    def test_int_output_equality(self, val):
        r = SolveResult(output=val)
        assert int(r) == val

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_float_output_equality(self, val):
        r = SolveResult(output=val)
        assert float(r) == val

    @given(st.text(min_size=1))
    def test_str_output_equality(self, val):
        r = SolveResult(output=val)
        assert str(r) == val

    @given(st.lists(st.integers(), max_size=10), st.integers())
    def test_list_contains_membership(self, data, val):
        r = SolveResult(output=data)
        assert (val in r) == (val in data)

    @given(st.lists(st.integers(), max_size=10), st.integers(min_value=0, max_value=9))
    def test_list_index_access(self, data, idx):
        assume(idx < len(data))
        r = SolveResult(output=data)
        assert r[idx] == data[idx]

    @given(st.lists(st.integers(), max_size=10), st.lists(st.integers(), max_size=10))
    def test_list_add(self, a, b):
        ra = SolveResult(output=a)
        rb = SolveResult(output=b)
        assert ra + b == a + b
        assert a + rb == a + b

    def test_error_result_not_ok(self):
        r = SolveResult(error="test error")
        assert not r.ok
        assert not r
        assert r.output is None

    def test_none_output_not_ok(self):
        r = SolveResult(output=None)
        assert not r.ok
        assert not r

    def test_empty_list_ok(self):
        r = SolveResult(output=[])
        assert r.ok

    def test_zero_output_ok(self):
        r = SolveResult(output=0)
        assert r.ok

    @given(st.dictionaries(st.text(), st.integers()))
    def test_dict_output_iteration_keys(self, data):
        r = SolveResult(output=data)
        assert list(r) == list(data.keys())

    @given(st.tuples(st.integers(), st.integers(), st.integers()))
    def test_tuple_output_iteration(self, data):
        r = SolveResult(output=data)
        assert list(r) == list(data)


class TestNormalizeProperties:
    @given(st.none())
    def test_none_normalizes_to_none(self, val):
        assert normalize(val) is None

    @given(st.booleans())
    def test_bool_unchanged(self, val):
        assert normalize(val) is val

    @given(st.integers())
    def test_int_unchanged(self, val):
        assert normalize(val) == val

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_float_unchanged(self, val):
        assert normalize(val) == val

    @given(st.text())
    def test_string_unchanged(self, val):
        assume(not val.strip().startswith(("/", "~", "http://", "https://", "{", "[")))
        assert normalize(val) == val

    @given(st.lists(st.integers(), max_size=50))
    def test_list_idempotent(self, data):
        assert normalize(normalize(data)) == normalize(data)

    @given(st.dictionaries(st.text(), st.integers(), max_size=10))
    def test_dict_idempotent(self, data):
        assert normalize(normalize(data)) == normalize(data)

    @given(st.lists(st.integers(), max_size=50))
    def test_normalize_preserves_list_values(self, data):
        result = normalize(data)
        assert result == data

    @given(st.dictionaries(st.text(), st.integers(), max_size=10))
    def test_normalize_preserves_dict_values(self, data):
        result = normalize(data)
        assert result == data

    @given(st.sets(st.integers(), max_size=50))
    def test_normalize_set_to_sorted_list(self, data):
        result = normalize(data)
        assert isinstance(result, list)
        assert result == sorted(list(data))

    @given(st.fractions())
    def test_normalize_fraction_to_float(self, val):
        result = normalize(val)
        assert isinstance(result, float)
        assert abs(result - float(val)) < 1e-10

    @given(st.decimals(allow_nan=False, allow_infinity=False))
    def test_normalize_decimal_to_float(self, val):
        result = normalize(val)
        assert isinstance(result, float)

    @given(st.complex_numbers(allow_nan=False, allow_infinity=False))
    def test_normalize_complex_to_dict(self, val):
        result = normalize(val)
        assert isinstance(result, dict)
        assert "real" in result
        assert "imag" in result
        assert abs(result["real"] - val.real) < 1e-10
        assert abs(result["imag"] - val.imag) < 1e-10


class TestDetectTypeProperties:
    @given(st.none())
    def test_detect_none(self, val):
        assert detect_type(val) == "none"

    @given(st.booleans())
    def test_detect_bool(self, val):
        assert detect_type(val) == "bool"

    @given(st.integers())
    def test_detect_int(self, val):
        assert detect_type(val) == "int"

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_detect_float(self, val):
        assert detect_type(val) == "float"

    @given(st.text())
    def test_detect_str(self, val):
        assert detect_type(val) == "str"

    @given(st.lists(st.integers(), max_size=100))
    def test_detect_list(self, val):
        result = detect_type(val)
        assert result.startswith("list(")
        assert str(len(val)) in result

    @given(st.dictionaries(st.text(), st.integers(), max_size=50))
    def test_detect_dict(self, val):
        result = detect_type(val)
        assert result.startswith("dict(")
        assert str(len(val)) in result

    @given(st.sets(st.integers(), max_size=50))
    def test_detect_set(self, val):
        result = detect_type(val)
        assert result.startswith("set(")
        assert str(len(val)) in result


class TestShortcutsProperties:
    @given(st.lists(st.integers(), min_size=1, max_size=50))
    def test_sort_preserves_elements(self, data):
        result = shortcuts.sort(data)
        assert sorted(data) == result
        assert len(result) == len(data)

    @given(st.lists(st.integers(), min_size=1, max_size=20), st.booleans())
    def test_sort_reverse(self, data, reverse):
        result = shortcuts.sort(data, reverse=reverse)
        expected = sorted(data, reverse=reverse)
        assert result == expected

    @given(st.lists(st.integers(), min_size=1, max_size=20), st.integers())
    def test_search_index_or_minus_one(self, data, target):
        idx = shortcuts.search(data, target)
        if target in data:
            assert idx == data.index(target)
        else:
            assert idx == -1

    @given(st.lists(st.integers(), min_size=1, max_size=20))
    def test_rank_length(self, data):
        result = shortcuts.rank(data)
        assert len(result) == len(data)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    @settings(deadline=None)
    @given(st.floats(min_value=-10, max_value=10), st.floats(min_value=0.1, max_value=10))
    def test_minimize_quadratic(self, center, scale):
        result = shortcuts.minimize(
            lambda x: scale * (x - center) ** 2,
            bounds=(center - 5, center + 5),
            steps=10000
        )
        assert abs(result - center) < 0.5

    @settings(deadline=None)
    @given(st.floats(min_value=-100, max_value=100))
    def test_maximize_negative_quadratic(self, center):
        result = shortcuts.maximize(
            lambda x: -(x - center) ** 2,
            bounds=(center - 50, center + 50),
            steps=10000
        )
        assert abs(result - center) < 1.0


class TestMindProperties:
    @settings(deadline=None)
    @given(st.lists(st.integers(), min_size=1, max_size=20))
    def test_mind_sort_any_list(self, data):
        tmp = tempfile.mkdtemp()
        m = Mind(tmp)
        r = m.solve("sort", data)
        assert r.output == sorted(data)
        assert r.ok

    @settings(deadline=None)
    @given(st.text(min_size=1))
    def test_mind_solve_never_crashes(self, problem_text):
        tmp = tempfile.mkdtemp()
        m = Mind(tmp)
        r = m.solve(problem_text, [1, 2, 3])
        assert isinstance(r, SolveResult)

    @settings(deadline=None)
    @given(st.lists(st.integers(), min_size=1, max_size=10))
    def test_mind_solve_count_increments(self, data):
        tmp = tempfile.mkdtemp()
        m = Mind(tmp)
        count = m._solve_count
        m.solve("sort", data)
        assert m._solve_count == count + 1

    @settings(deadline=None)
    @given(st.integers(min_value=1, max_value=100))
    def test_mind_train_any_epochs(self, epochs):
        tmp = tempfile.mkdtemp()
        m = Mind(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = m.train(epochs=epochs)
            assert "status" in result
