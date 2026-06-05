import pytest
from pathlib import Path
from aalgoi.core.mind import create_mind
from aalgoi.core.mind.solving_loop import UniversalSolution


class TestCreateMind:
    def test_create_mind_with_defaults(self):
        loop = create_mind()
        assert loop is not None
        assert hasattr(loop, "solve")

    def test_create_mind_invokes_solve(self):
        loop = create_mind()
        solution = loop.solve("sort this array", [3, 1, 2], max_iterations=5)
        assert isinstance(solution, UniversalSolution)
        assert solution.solve_time_ms > 0
        assert solution.iterations >= 0

    def test_solve_with_different_problems(self):
        loop = create_mind()

        sol1 = loop.solve("sort", [3, 1, 2], max_iterations=5)
        assert sol1.solve_time_ms > 0

        sol2 = loop.solve("find shortest path", {"edges": [(1, 2, 5)]}, max_iterations=5)
        assert sol2.solve_time_ms > 0

    def test_solve_returns_tracked_actions(self):
        loop = create_mind()
        solution = loop.solve("test", [1], max_iterations=5)
        assert isinstance(solution.actions_taken, list)
