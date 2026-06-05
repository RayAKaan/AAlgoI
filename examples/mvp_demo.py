#!/usr/bin/env python3
"""3-Domain MVP demo: sorting, pathfinding, optimization with UniversalSolver."""

import time
from pprint import pprint

from aalgoi.algorithms.sorting import QuickSort, MergeSort
from aalgoi.algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder
from aalgoi.algorithms.optimization import GreedyKnapsack, SimulatedAnnealing
from aalgoi.core.problem_spec import ProblemSpec, ProblemType, Objective


def demo_sorting():
    print("=" * 60)
    print("DOMAIN 1: SORTING")
    print("=" * 60)
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    for algo in [QuickSort(), MergeSort()]:
        start = time.perf_counter()
        result = algo.process(data.copy())
        elapsed = (time.perf_counter() - start) * 1000
        ok = result == sorted(data)
        print(f"  {algo.name:20s} {str(result):50s} {elapsed:6.2f}ms {'OK' if ok else 'FAIL'}")


def demo_pathfinding():
    print()
    print("=" * 60)
    print("DOMAIN 2: PATHFINDING")
    print("=" * 60)
    graph = {
        "A": {"B": 1, "C": 4},
        "B": {"C": 2, "D": 5, "E": 1},
        "C": {"D": 1},
        "D": {},
        "E": {"D": 2},
    }
    print(f"  Graph: A -> B(1), C(4); B -> C(2), D(5), E(1); C -> D(1); E -> D(2)")
    for algo in [Dijkstra(), AStar(), BFSPathfinder()]:
        start = time.perf_counter()
        if algo.name == "bfs_path":
            result = algo.process({"graph": graph, "start": "A", "end": "D"})
        else:
            result = algo.process({"graph": graph, "start": "A", "end": "D"})
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {algo.name:20s} A->D: {str(result):30s} {elapsed:6.2f}ms")


def demo_optimization():
    print()
    print("=" * 60)
    print("DOMAIN 3: OPTIMIZATION")
    print("=" * 60)
    items = [
        {"value": 60, "weight": 10},
        {"value": 100, "weight": 20},
        {"value": 120, "weight": 30},
    ]
    capacity = 50
    print(f"  Items: {items}")
    print(f"  Capacity: {capacity}")

    for algo in [GreedyKnapsack(), SimulatedAnnealing()]:
        start = time.perf_counter()
        if algo.name == "simulated_annealing":
            result = algo.process({"items": items, "capacity": capacity, "iterations": 500})
        else:
            result = algo.process({"items": items, "capacity": capacity})
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {algo.name:20s} value={result['value']:6.1f} weight={result['weight']:4.0f}  {elapsed:8.2f}ms")


def demo_universal_solver():
    print()
    print("=" * 60)
    print("UNIVERSAL SOLVER (auto-detect + auto-wrap)")
    print("=" * 60)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()

    spec = ProblemSpec(name="sort numbers", problem_type=ProblemType.TRANSFORMATION)
    result = solver.solve(spec, [3, 1, 4, 1, 5], expected=[1, 1, 3, 4, 5])
    print(f"  Sorting: success={result['success']} pipeline={result['pipeline']}")
    print(f"           result={result['result']}")
    print(f"           time={result['time_ms']:.2f}ms")


if __name__ == "__main__":
    demo_sorting()
    demo_pathfinding()
    demo_optimization()
    demo_universal_solver()
