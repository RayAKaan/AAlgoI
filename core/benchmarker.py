"""
Benchmarking Framework
Compare AAlgoI against standard implementations.
"""

import time
import logging
from typing import Dict, Any

from core.problem_spec import ProblemSpec, ProblemType

logger = logging.getLogger(__name__)


class Benchmarker:
    """
    Compare AAlgoI against standard library implementations.
    Provides speedup factors and performance analysis.
    """

    def __init__(self, solver):
        self.solver = solver

    def compare(self, spec: ProblemSpec, data: Any) -> Dict[str, Any]:
        """
        Run comparison benchmark: AAlgoI vs standard library.
        Returns timing, speedup, winner.
        """
        aalgoi_time, aalgoi_algo = self._run_aalgoi(spec, data)
        baseline_time, baseline_name = self._run_baseline(spec, data)

        speedup = baseline_time / aalgoi_time if aalgoi_time > 0 else float("inf")

        return {
            "aalgoi_time_ms": round(aalgoi_time * 1000, 2),
            "baseline_time_ms": round(baseline_time * 1000, 2),
            "speedup_factor": round(speedup, 2),
            "aalgoi_algorithm": aalgoi_algo,
            "baseline_algorithm": baseline_name,
            "winner": "AAlgoI" if speedup > 1.05 else "Baseline",
        }

    def _run_aalgoi(self, spec: ProblemSpec, data: Any) -> tuple:
        """Run AAlgoI solver and return (time_seconds, algorithm_name)."""
        start = time.time()
        result = self.solver.solve(spec, data)
        elapsed = time.time() - start
        return elapsed, result.get("algorithm", "unknown")

    def _run_baseline(self, spec: ProblemSpec, data: Any) -> tuple:
        """Run standard library implementation and return (time_seconds, name)."""
        start = time.time()
        name = "stdlib"

        if spec.problem_type == ProblemType.SORTING:
            name = "python_timsort"
            if isinstance(data, list):
                _ = sorted(data)

        elif spec.problem_type == ProblemType.PATHFINDING:
            name = "networkx_dijkstra"
            try:
                import networkx as nx
                if isinstance(data, dict) and "graph" in data:
                    G = nx.DiGraph(data["graph"])
                    _ = nx.dijkstra_path(G, data.get("start"), data.get("end"))
            except (ImportError, Exception):
                pass

        elif spec.problem_type == ProblemType.OPTIMIZATION:
            name = "greedy_heuristic"
            if isinstance(data, dict) and "items" in data and "capacity" in data:
                items = sorted(
                    data["items"],
                    key=lambda x: x.get("value", 0) / max(x.get("weight", 1), 1),
                    reverse=True,
                )
                total_w = 0
                selected = []
                for item in items:
                    if total_w + item.get("weight", 0) <= data["capacity"]:
                        selected.append(item)
                        total_w += item.get("weight", 0)

        elapsed = time.time() - start
        return elapsed, name
