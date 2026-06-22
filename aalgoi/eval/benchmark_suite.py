from __future__ import annotations

import random
import time
from typing import Any

from aalgoi.algorithms.registry import AlgorithmRegistry, get_registry
from aalgoi.problems.generators import generate_example
from aalgoi.problems.oracles import evaluate as oracle_evaluate
from aalgoi.selection.planner import Planner
from aalgoi.types import BenchmarkReport, ProblemSpec, ProblemTask


class BenchmarkSuite:
    def __init__(self, name: str = "core-v1", planner: Planner | None = None) -> None:
        self.name = name
        self.planner = planner or Planner()
        self._problems: list[ProblemSpec] = []

    def add(self, spec: ProblemSpec) -> None:
        self._problems.append(spec)

    def add_generated(self, task: ProblemTask, count: int = 5, rng: random.Random | None = None) -> None:
        for _ in range(count):
            inputs, expected = generate_example(task, rng=rng)
            spec = ProblemSpec(
                id=f"{task.value}_{len(self._problems)}",
                task=task,
                domain=task.domain if hasattr(task, "domain") else None,
                inputs=inputs,
            )
            self._problems.append(spec)

    def run(self) -> BenchmarkReport:
        t0 = time.time()
        correct = 0
        failed = 0
        errors = 0
        by_task: dict[str, dict] = {}
        regressions: list[str] = []

        for spec in self._problems:
            try:
                desc = self._task_to_text(spec.task)
                result = self.planner.solve(desc, spec.inputs)
                if result.ok:
                    passed = oracle_evaluate(spec.task, spec.inputs, result.output) if result.output is not None else False
                else:
                    passed = False
                if passed:
                    correct += 1
                else:
                    failed += 1
                task_key = spec.task.value
                if task_key not in by_task:
                    by_task[task_key] = {"total": 0, "passed": 0, "failed": 0}
                by_task[task_key]["total"] += 1
                if passed:
                    by_task[task_key]["passed"] += 1
                else:
                    by_task[task_key]["failed"] += 1
            except Exception:
                errors += 1

        total = len(self._problems)
        elapsed = (time.time() - t0) * 1000
        accuracy = correct / max(total, 1)

        return BenchmarkReport(
            suite=self.name,
            total=total,
            correct=correct,
            failed=failed,
            errors=errors,
            accuracy=accuracy,
            by_task=by_task,
            regressions=regressions,
            time_ms=elapsed,
        )

    @staticmethod
    def _task_to_text(task: ProblemTask) -> str:
        mapping = {
            "sort": "sort the list",
            "counting_sort": "counting sort the list",
            "linear_search": "linear search for target",
            "binary_search": "binary search for target",
            "two_sum": "find two sum target",
            "lower_bound": "find lower bound of target",
            "gcd": "compute gcd",
            "lcm": "compute lcm",
            "is_prime": "check if prime",
            "sieve": "sieve of eratosthenes",
            "fast_exponentiation": "fast exponentiation",
            "fibonacci": "fibonacci number",
            "palindrome": "check palindrome",
            "anagram": "check anagram",
            "kmp": "kmp pattern search",
            "rabin_karp": "rabin karp pattern search",
            "edit_distance": "edit distance",
            "lcs": "longest common subsequence",
            "bfs": "bfs on graph",
            "dfs": "dfs on graph",
            "shortest_path_unweighted": "shortest path unweighted",
            "shortest_path_weighted": "shortest path weighted dijkstra",
            "shortest_path_negative": "shortest path negative weight bellman ford",
            "topological_sort": "topological sort",
            "cycle_detection": "cycle detection in graph",
            "connected_components": "connected components of graph",
            "mst": "minimum spanning tree",
            "max_flow": "maximum flow",
            "kadane": "kadane maximum subarray",
            "knapsack_01": "01 knapsack",
            "coin_change": "coin change",
            "lis": "longest increasing subsequence",
            "knapsack_fractional": "fractional knapsack",
        }
        return mapping.get(task.value, task.value)


def core_v1() -> BenchmarkSuite:
    suite = BenchmarkSuite("core-v1")
    rng = random.Random(1337)
    for task in ProblemTask:
        suite.add_generated(task, count=3, rng=rng)
    return suite
