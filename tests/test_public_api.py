"""
Tests for the public aalgoi API.
All calls validated against real core/mind/ APIs.
"""

from pathlib import Path

import pytest

import aalgoi
from aalgoi._core import AlgorithmInfo, BenchmarkReport, Mind
from aalgoi._data import detect_type, normalize
from aalgoi._result import SolveResult
from aalgoi._status import box, progress_bar, table


@pytest.fixture
def mind_path(tmp_path):
    return tmp_path / "test_mind"


@pytest.fixture
def mind(mind_path):
    return Mind(mind_path)


# ═══════════════════════════════════════════════════════════════
#  SolveResult
# ═══════════════════════════════════════════════════════════════

class TestSolveResult:
    def test_str_returns_output(self):
        r = SolveResult(output=[1, 2, 3])
        assert str(r) == "[1, 2, 3]"

    def test_repr_is_boxed(self):
        r = SolveResult(output=42, algorithm="gcd", complexity="O(1)", confidence=0.95)
        repr_str = repr(r)
        assert "╔" in repr_str
        assert "╚" in repr_str
        assert "gcd" in repr_str

    def test_repr_error(self):
        assert "❌" in repr(SolveResult(error="something failed"))

    def test_repr_no_output(self):
        assert "⚠" in repr(SolveResult(output=None))

    def test_repr_novel(self):
        r = SolveResult(output=42, is_novel=True, confidence=0.9)
        assert "🆕" in repr(r)

    def test_bool_true(self):
        assert bool(SolveResult(output=42)) is True

    def test_bool_false_none(self):
        assert bool(SolveResult(output=None)) is False

    def test_bool_false_error(self):
        assert bool(SolveResult(output=None, error="failed")) is False

    def test_eq_with_raw(self):
        r = SolveResult(output=[1, 2, 3])
        assert r == [1, 2, 3]

    def test_eq_with_result(self):
        assert SolveResult(output=42) == SolveResult(output=42)

    def test_ne(self):
        assert SolveResult(output=1) != SolveResult(output=2)

    def test_iter(self):
        assert list(SolveResult(output=[1, 2, 3])) == [1, 2, 3]

    def test_iter_none(self):
        assert list(SolveResult(output=None)) == []

    def test_getitem(self):
        r = SolveResult(output=[10, 20, 30])
        assert r[0] == 10
        assert r[2] == 30

    def test_len(self):
        assert len(SolveResult(output=[1, 2, 3, 4])) == 4

    def test_len_none(self):
        assert len(SolveResult(output=None)) == 0

    def test_contains(self):
        r = SolveResult(output=[1, 2, 3])
        assert 2 in r
        assert 5 not in r

    def test_add(self):
        r = SolveResult(output=[1, 2])
        assert r + [3] == [1, 2, 3]

    def test_radd(self):
        r = SolveResult(output=[2])
        assert [1] + r == [1, 2]

    def test_int(self):
        assert int(SolveResult(output=42)) == 42

    def test_float(self):
        assert float(SolveResult(output=3.14)) == pytest.approx(3.14)

    def test_ok_property(self):
        assert SolveResult(output=42).ok is True
        assert SolveResult(output=None).ok is False
        assert SolveResult(error="x").ok is False

    def test_all_properties(self):
        r = SolveResult(
            output=42, code="def solve(): return 42",
            algorithm="test_algo", complexity="O(1)",
            principle="optimal_substructure", time_ms=12.5,
            is_novel=True, confidence=0.95, iterations=7,
        )
        assert r.output == 42
        assert r.code == "def solve(): return 42"
        assert r.algorithm == "test_algo"
        assert r.complexity == "O(1)"
        assert r.principle == "optimal_substructure"
        assert r.time_ms == 12.5
        assert r.is_novel is True
        assert r.confidence == 0.95
        assert r.iterations == 7
        assert r.error is None

    def test_explain(self):
        r = SolveResult(
            output=42, algorithm="gcd",
            principle="divide_conquer", complexity="O(log n)", confidence=0.95,
        )
        explanation = r.explain()
        assert "gcd" in explanation
        assert "divide_conquer" in explanation
        assert "O(log n)" in explanation
        assert "95%" in explanation

    def test_explain_error(self):
        assert "timeout" in SolveResult(error="timeout").explain()

    def test_explain_novel(self):
        assert "🆕" in SolveResult(output=42, is_novel=True, confidence=0.9).explain()

    def test_explain_with_code(self):
        r = SolveResult(output=42, code="def solve():\n    return 42")
        assert "def solve" in r.explain()

    def test_hash(self):
        assert hash(SolveResult(output=42)) == hash(42)


# ═══════════════════════════════════════════════════════════════
#  Box / Table / Progress
# ═══════════════════════════════════════════════════════════════

class TestBox:
    def test_basic(self):
        result = box(["hello", "world"])
        assert "╔" in result and "╚" in result and "hello" in result

    def test_title(self):
        assert "Test" in box(["line1"], title="Test")

    def test_empty(self):
        assert box([]) == ""

    def test_alignment(self):
        result = box(["short", "a bit longer"])
        for line in result.split("\n")[1:-1]:
            assert line.startswith("║") and line.endswith("║")


class TestTable:
    def test_basic(self):
        result = table(["A", "B"], [["1", "2"], ["3", "4"]])
        assert "┌" in result and "├" in result and "A" in result

    def test_empty(self):
        assert table([], []) == ""


class TestProgressBar:
    def test_basic(self):
        result = progress_bar(50, 100, label="Training")
        assert "50%" in result and "█" in result and "░" in result

    def test_zero(self):
        assert "100%" in progress_bar(0, 0)


# ═══════════════════════════════════════════════════════════════
#  Mind class
# ═══════════════════════════════════════════════════════════════

class TestMind:
    def test_init_creates_directory(self, mind_path):
        Mind(mind_path)
        assert mind_path.exists()

    def test_path_property(self, mind):
        assert isinstance(mind.path, Path)

    def test_lazy_loading(self, mind):
        assert mind._loaded is False

    def test_ensure_loaded_uses_create_mind(self, mind):
        mind._ensure_loaded()
        assert mind._loaded is True
        assert mind._mind is not None

    def test_status(self, mind):
        status = mind.status()
        assert "Algorithms" in status
        assert "🧠" in status

    def test_algorithms_property(self, mind):
        algos = mind.algorithms
        assert isinstance(algos, dict)
        assert len(algos) > 0
        assert "quick_sort" in algos

    def test_algorithm_info(self, mind):
        info = mind.algorithms["quick_sort"]
        assert isinstance(info, AlgorithmInfo)
        assert info.name == "quick_sort"
        assert info.time_complexity is not None

    def test_algorithm_info_repr(self, mind):
        assert "quick_sort" in repr(mind.algorithms["quick_sort"])

    def test_algorithm_info_display(self, mind):
        display = mind.algorithms["quick_sort"].display()
        assert "quick_sort" in display
        assert "📋" in display

    def test_principles_property(self, mind):
        principles = mind.principles
        assert isinstance(principles, list)
        assert "optimal_substructure" in principles
        assert "greedy_exchange" in principles

    def test_problems_property(self, mind):
        problems = mind.problems
        assert isinstance(problems, list)
        assert "SORTING" in problems
        assert "PATHFINDING" in problems


class TestMindSolve:
    def test_solve_sort(self, mind):
        result = mind.solve("sort the array", [3, 1, 4, 1, 5])
        assert isinstance(result, SolveResult)
        if result.output is not None:
            assert result.output == [1, 1, 3, 4, 5]

    def test_solve_returns_result(self, mind):
        result = mind.solve("find maximum sum subarray", [-2, 1, -3, 4, -1, 2, 1, -5, 4])
        assert isinstance(result, SolveResult)
        assert result.time_ms >= 0

    def test_solve_with_dict_data(self, mind):
        result = mind.solve(
            "find two numbers that sum to target",
            {"nums": [2, 7, 11, 15], "target": 9},
        )
        assert isinstance(result, SolveResult)

    def test_solve_with_none_data(self, mind):
        result = mind.solve("sort the array", None)
        assert isinstance(result, SolveResult)

    def test_solve_transparent_result(self, mind):
        result = mind.solve("sort the array", [3, 1, 2])
        if isinstance(result.output, list):
            assert len(result) == 3
            assert result[0] == 1

    def test_solve_normalizes_data(self, mind):
        result = mind.solve("sort", (3, 1, 2))
        assert isinstance(result, SolveResult)

    def test_solve_tracks_locally(self, mind):
        mind.solve("sort", [1, 2, 3])
        mind.solve("sort", [3, 2, 1])
        assert mind._solve_count == 2


class TestMindCheckpoint:
    def test_checkpoint_returns_path_or_none(self, mind):
        result = mind.checkpoint()
        assert result is None or isinstance(result, str)

    def test_rollback_returns_dict(self, mind):
        result = mind.rollback("last_good")
        assert isinstance(result, dict)
        assert "success" in result


class TestMindFederation:
    def test_share_returns_int(self, mind):
        result = mind.share()
        assert isinstance(result, int)
        assert result >= 0

    def test_receive_returns_dict(self, mind):
        result = mind.receive()
        assert isinstance(result, dict)
        assert "updates_processed" in result
        assert "algorithms_imported" in result


# ═══════════════════════════════════════════════════════════════
#  Session
# ═══════════════════════════════════════════════════════════════

class TestMindSession:
    def test_session_context_manager(self, mind_path):
        with aalgoi.session(mind_path) as m:
            r = m.solve("sort", [3, 1, 2])
            assert isinstance(r, SolveResult)

    def test_session_learn(self, mind_path):
        with aalgoi.session(mind_path) as m:
            r = m.learn("sort", [3, 1, 2], expected=[1, 2, 3])
            assert isinstance(r, SolveResult)

    def test_session_status(self, mind_path):
        with aalgoi.session(mind_path) as m:
            status = m.status()
            assert "Session" in status


# ═══════════════════════════════════════════════════════════════
#  BenchmarkReport
# ═══════════════════════════════════════════════════════════════

class TestBenchmarkReport:
    def test_creation(self):
        data = {
            "total": 50, "correct": 42, "failed": 8, "errors": 0,
            "accuracy": 0.84,
            "by_domain": {"integers": {"correct": 5, "total": 5}},
            "problems": [],
        }
        report = BenchmarkReport(data)
        assert report.accuracy == 0.84
        assert report.correct == 42
        assert report.total == 50

    def test_repr(self):
        data = {
            "total": 10, "correct": 8, "failed": 2, "errors": 0,
            "accuracy": 0.8,
            "by_domain": {"integers": {"correct": 4, "total": 5}},
            "problems": [],
        }
        report = BenchmarkReport(data)
        assert "📊" in repr(report)
        assert "80%" in repr(report)

    def test_details(self):
        data = {
            "total": 2, "correct": 1, "accuracy": 0.5,
            "by_domain": {},
            "problems": [{"id": "t1"}, {"id": "t2"}],
        }
        report = BenchmarkReport(data)
        assert len(report.details()) == 2


# ═══════════════════════════════════════════════════════════════
#  Data normalizer
# ═══════════════════════════════════════════════════════════════

class TestNormalize:
    def test_none(self):
        assert normalize(None) is None

    def test_int(self):
        assert normalize(42) == 42

    def test_float(self):
        assert normalize(3.14) == 3.14

    def test_str(self):
        assert normalize("hello") == "hello"

    def test_bool(self):
        assert normalize(True) is True

    def test_list(self):
        assert normalize([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert normalize((1, 2, 3)) == [1, 2, 3]

    def test_set(self):
        result = normalize({3, 1, 2})
        assert sorted(result) == [1, 2, 3]

    def test_dict(self):
        assert normalize({"a": 1}) == {"a": 1}

    def test_nested(self):
        assert normalize({"nums": [1, 2], "target": 5}) == {"nums": [1, 2], "target": 5}


class TestDetectType:
    def test_list(self):
        assert "list" in detect_type([1, 2, 3])

    def test_dict(self):
        assert "dict" in detect_type({"a": 1})

    def test_int(self):
        assert detect_type(42) == "int"

    def test_none(self):
        assert detect_type(None) == "none"


# ═══════════════════════════════════════════════════════════════
#  Module-level API
# ═══════════════════════════════════════════════════════════════

class TestModuleAPI:
    def test_version(self):
        assert aalgoi.__version__ == "2.2.1"

    def test_solve_exposed(self):
        assert callable(aalgoi.solve)

    def test_session_exposed(self):
        assert callable(aalgoi.session)

    def test_mind_class_exposed(self):
        assert aalgoi.Mind is Mind

    def test_solve_result_exposed(self):
        assert aalgoi.SolveResult is SolveResult

    def test_normalize_exposed(self):
        assert callable(aalgoi.normalize)

    def test_detect_type_exposed(self):
        assert callable(aalgoi.detect_type)

    def test_all_complete(self):
        for name in ["solve", "session", "Mind", "SolveResult",
                      "normalize", "detect_type", "__version__"]:
            assert name in aalgoi.__all__, f"{name} missing from __all__"
