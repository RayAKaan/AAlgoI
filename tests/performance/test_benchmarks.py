"""Basic performance benchmarks to detect regressions."""

import pytest
import time


@pytest.mark.performance
class TestSortingPerformance:

    def test_sorting_large_list(self):
        from aalgoi.pipeline import UniversalSolver
        from aalgoi.core.problem_spec import ProblemSpec, ProblemType
        solver = UniversalSolver()
        spec = ProblemSpec(name="bench_sort", problem_type=ProblemType.SORTING)
        data = list(range(10000, 0, -1))
        start = time.perf_counter()
        result = solver.solve(spec, data)
        elapsed = time.perf_counter() - start
        assert result["success"]
        assert result["result"] == list(range(1, 10001))
        assert elapsed < 10.0, f"Sorting 10k items took {elapsed:.2f}s (limit 10s)"


@pytest.mark.performance
class TestOraclePerformance:

    def test_oracle_lookup_speed(self):
        from aalgoi.core.oracles import evaluate
        from aalgoi.core.problem_spec import ProblemType
        import time
        start = time.perf_counter()
        for _ in range(1000):
            evaluate(ProblemType.SORTING, [3, 1, 2], [1, 2, 3])
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"1000 oracle lookups took {elapsed:.2f}s (limit 2s)"
