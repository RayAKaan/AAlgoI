"""Comprehensive tests for SolveResult class."""
import pytest
from aalgoi._result import SolveResult


class TestSolveResultBasic:
    def test_successful_result(self):
        r = SolveResult(output=[1, 2, 3])
        assert r.ok
        assert r.success
        assert bool(r)
        assert r.error is None

    def test_error_result(self):
        r = SolveResult(error="test error")
        assert not r.ok
        assert not r.success
        assert not bool(r)
        assert r.error == "test error"

    def test_none_output(self):
        r = SolveResult(output=None)
        assert not r.ok
        assert not r.success

    def test_empty_list_output(self):
        r = SolveResult(output=[])
        assert r.ok
        assert bool(r)

    def test_zero_output(self):
        r = SolveResult(output=0)
        assert r.ok
        assert bool(r)

    def test_empty_string_output(self):
        r = SolveResult(output="")
        assert r.ok

    def test_false_output(self):
        r = SolveResult(output=False)
        assert r.ok
        assert bool(r)


class TestSolveResultAttributes:
    def test_all_attributes(self):
        r = SolveResult(
            output=[1, 2, 3],
            code="def solve(): return [1, 2, 3]",
            algorithm="my_algo",
            complexity="O(n)",
            principle="test",
            time_ms=100.0,
            is_novel=True,
            confidence=0.95,
            iterations=5,
        )
        assert r.output == [1, 2, 3]
        assert r.code == "def solve(): return [1, 2, 3]"
        assert r.algorithm == "my_algo"
        assert r.complexity == "O(n)"
        assert r.principle == "test"
        assert r.time_ms == 100.0
        assert r.is_novel is True
        assert r.confidence == 0.95
        assert r.iterations == 5

    def test_default_values(self):
        r = SolveResult()
        assert r.output is None
        assert r.code is None
        assert r.algorithm is None
        assert r.complexity is None
        assert r.principle is None
        assert r.time_ms == 0.0
        assert r.is_novel is False
        assert r.confidence == 0.0
        assert r.iterations == 0
        assert r.error is None


class TestSolveResultStringRepresentation:
    def test_str_with_output(self):
        r = SolveResult(output=[1, 2, 3])
        assert str(r) == "[1, 2, 3]"

    def test_str_with_error(self):
        r = SolveResult(error="fail")
        assert "fail" in str(r)

    def test_repr_with_output(self):
        r = SolveResult(output=[1, 2, 3], algorithm="test", confidence=0.9)
        repr_str = repr(r)
        assert "test" in repr_str
        assert "SolveResult" in repr_str

    def test_repr_with_error(self):
        r = SolveResult(error="fail")
        repr_str = repr(r)
        assert "fail" in repr_str

    def test_explain_success(self):
        r = SolveResult(
            output=[1, 2, 3],
            algorithm="test_algo",
            complexity="O(n)",
            principle="test",
            confidence=0.95,
            time_ms=100.0,
            iterations=5,
        )
        explanation = r.explain()
        assert "test_algo" in explanation
        assert "O(n)" in explanation
        assert "95%" in explanation
        assert "100.0ms" in explanation
        assert "5" in explanation

    def test_explain_novel(self):
        r = SolveResult(output=[1, 2, 3], algorithm="novel_algo", is_novel=True)
        explanation = r.explain()
        assert "Novel" in explanation

    def test_explain_with_code(self):
        r = SolveResult(
            output=[1, 2, 3],
            algorithm="test",
            code="def test(): pass",
        )
        explanation = r.explain()
        assert "def test(): pass" in explanation

    def test_explain_error(self):
        r = SolveResult(error="fail")
        explanation = r.explain()
        assert "fail" in explanation


class TestSolveResultContainerOperations:
    def test_iteration_list(self):
        r = SolveResult(output=[1, 2, 3])
        assert list(r) == [1, 2, 3]

    def test_iteration_dict(self):
        r = SolveResult(output={"a": 1, "b": 2})
        assert list(r) == ["a", "b"]

    def test_iteration_empty(self):
        r = SolveResult(output=[])
        assert list(r) == []

    def test_iteration_none(self):
        r = SolveResult(output=None)
        assert list(r) == []

    def test_indexing(self):
        r = SolveResult(output=[10, 20, 30])
        assert r[0] == 10
        assert r[1] == 20
        assert r[-1] == 30

    def test_indexing_none(self):
        r = SolveResult(output=None)
        with pytest.raises(IndexError):
            _ = r[0]

    def test_len_list(self):
        r = SolveResult(output=[1, 2, 3])
        assert len(r) == 3

    def test_len_dict(self):
        r = SolveResult(output={"a": 1, "b": 2})
        assert len(r) == 2

    def test_len_empty(self):
        r = SolveResult(output=[])
        assert len(r) == 0

    def test_len_none(self):
        r = SolveResult(output=None)
        assert len(r) == 0

    def test_contains(self):
        r = SolveResult(output=[1, 2, 3])
        assert 2 in r
        assert 5 not in r

    def test_contains_none(self):
        r = SolveResult(output=None)
        assert 1 not in r

    def test_contains_dict(self):
        r = SolveResult(output={"a": 1, "b": 2})
        assert "a" in r
        assert "c" not in r


class TestSolveResultArithmetic:
    def test_add_list(self):
        r = SolveResult(output=[1, 2])
        assert r + [3, 4] == [1, 2, 3, 4]

    def test_radd_list(self):
        r = SolveResult(output=[3, 4])
        assert [1, 2] + r == [1, 2, 3, 4]

    def test_add_none(self):
        r = SolveResult(output=None)
        try:
            result = r + [1]
            assert result is NotImplemented
        except TypeError:
            pass

    def test_radd_none(self):
        r = SolveResult(output=None)
        try:
            result = [1] + r
            assert result is NotImplemented
        except TypeError:
            pass


class TestSolveResultTypeConversion:
    def test_int_conversion(self):
        r = SolveResult(output=42)
        assert int(r) == 42

    def test_float_conversion(self):
        r = SolveResult(output=3.14)
        assert float(r) == pytest.approx(3.14)

    def test_int_list_falls_back(self):
        r = SolveResult(output=[1, 2, 3])
        assert int(r) == 0

    def test_float_list_falls_back(self):
        r = SolveResult(output=[1, 2, 3])
        assert float(r) == 0.0

    def test_int_string_falls_back(self):
        r = SolveResult(output="abc")
        assert int(r) == 0


class TestSolveResultHash:
    def test_hash_int(self):
        r = SolveResult(output=42)
        assert hash(r) == hash(42)

    def test_hash_string(self):
        r = SolveResult(output="hello")
        assert hash(r) == hash("hello")

    def test_hash_tuple(self):
        r = SolveResult(output=(1, 2, 3))
        assert hash(r) == hash((1, 2, 3))

    def test_hash_list_default(self):
        r = SolveResult(output=[1, 2, 3])
        hash(r)

    def test_hash_dict_default(self):
        r = SolveResult(output={"a": 1})
        hash(r)


class TestSolveResultEquality:
    def test_equal_outputs(self):
        r1 = SolveResult(output=[1, 2, 3], algorithm="a")
        r2 = SolveResult(output=[1, 2, 3], algorithm="b")
        assert r1 == r2

    def test_equal_to_output(self):
        r = SolveResult(output=[1, 2, 3])
        assert r == [1, 2, 3]

    def test_not_equal(self):
        r1 = SolveResult(output=[1, 2, 3])
        r2 = SolveResult(output=[3, 2, 1])
        assert r1 != r2

    def test_not_equal_to_other(self):
        r = SolveResult(output=[1, 2, 3])
        assert r != [3, 2, 1]
