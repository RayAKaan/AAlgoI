"""Quick smoke tests for the aalgoi public API."""

from importlib.util import find_spec

import pytest

import aalgoi


def test_version():
    assert aalgoi.__version__ == "2.2.1"


def test_solve_sort():
    result = aalgoi.solve("sort the array", [3, 1, 2])
    assert result.output == [1, 2, 3]
    assert result.algorithm == "tim_sort"


def test_solve_search():
    result = aalgoi.solve("binary search target", {"nums": [1, 3, 5, 7, 9], "target": 5})
    assert result.output == 2


def test_mind_basic():
    m = aalgoi.Mind("~/.aalgoi/mind")
    result = m.solve("sort", [3, 1, 2])
    assert result.ok
    assert result.output == [1, 2, 3]


def test_mind_algorithms():
    m = aalgoi.Mind("~/.aalgoi/mind")
    assert len(m.algorithms) >= 10
    assert len(m.principles) >= 5


def test_session():
    with aalgoi.session() as s:
        r = s.solve("sort", [3, 1, 2])
        assert r.output == [1, 2, 3]
        status = s.status()
        assert "Solved" in status


def test_normalize():
    assert aalgoi.normalize([1, 2, 3]) == [1, 2, 3]
    assert aalgoi.normalize(42) == 42


def test_shortcuts():
    from aalgoi.shortcuts import sort, search
    assert sort([3, 1, 2]) == [1, 2, 3]
    assert search([1, 2, 3], 2) == 1


def test_public_types():
    assert aalgoi.AlgorithmInfo is not None
    assert aalgoi.BenchmarkReport is not None
    assert aalgoi.SolveResult is not None


def test_solve_empty_data():
    result = aalgoi.solve("do something", None)
    assert result.output is None


def test_solve_pathfinding():
    result = aalgoi.solve(
        "find shortest path",
        {"graph": [(0, 1), (1, 2)], "start": 0, "end": 2},
    )
    assert result.ok


def test_word_boundary_no_false_positive():
    result = aalgoi.solve("transportation issues", [1, 2, 3])
    assert result.output is None


torch_missing = pytest.mark.skipif(
    find_spec("torch") is None, reason="requires torch"
)


def test_mind_benchmark():
    m = aalgoi.Mind("~/.aalgoi/mind")
    report = m.benchmark()
    assert isinstance(report, aalgoi.BenchmarkReport)
