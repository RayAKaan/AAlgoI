import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem_spec import ProblemSpec, ProblemType
from pipeline import UniversalSolver


class TestNeverFail:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_identity_algorithm_always_works(self, solver):
        assert "identity" in solver.registry
        identity = solver.registry["identity"]
        result = identity.process("anything")
        assert result == "anything"
        assert identity.validate_output("anything", result) is True

    def test_identity_on_none(self, solver):
        identity = solver.registry["identity"]
        result = identity.process(None)
        assert result is None

    def test_safe_sort_on_list(self, solver):
        safe_sort = solver.registry["safe_sort"]
        result = safe_sort.process([3, 1, 2])
        assert result == [1, 2, 3]

    def test_safe_sort_on_bad_data(self, solver):
        safe_sort = solver.registry["safe_sort"]
        result = safe_sort.process([{"a": 1}, {"b": 2}])
        assert result is not None

    def test_safe_path_on_invalid_graph(self, solver):
        safe_path = solver.registry["safe_path"]
        result = safe_path.process({"graph": {}, "start": "a", "end": "b"})
        assert result == []

    def test_safe_path_on_non_dict(self, solver):
        safe_path = solver.registry["safe_path"]
        result = safe_path.process(None)
        assert result == []

    def test_safe_knapsack_on_empty(self, solver):
        safe_knap = solver.registry["safe_knapsack"]
        result = safe_knap.process({"items": [], "capacity": 0})
        assert result["selected"] == []

    def test_safe_knapsack_on_bad_data(self, solver):
        safe_knap = solver.registry["safe_knapsack"]
        result = safe_knap.process("garbage")
        assert result["selected"] == []

    def test_prepare_input_data_none(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver._prepare_input_data(spec, None)
        assert result == []

    def test_prepare_input_data_pathfinding_none(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        result = solver._prepare_input_data(spec, None)
        assert isinstance(result, dict)
        assert "graph" in result

    def test_prepare_input_data_string(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver._prepare_input_data(spec, "[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_prepare_input_data_string_not_json(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver._prepare_input_data(spec, "hello")
        assert result == "hello"

    def test_solve_none_data_returns_identity(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, None)
        assert result["success"] is True
        assert result["algorithm"] != ""

    def test_solve_string_data(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, "[3, 1, 2]")
        assert result["success"] is True

    def test_context_engine_fallback_for_unknown(self, solver):
        from core.context_engine import ContextEngine
        engine = ContextEngine()
        context = engine.analyze([1, 2, 3], task_type="unknown")
        assert context["task_type"] == "safety"
