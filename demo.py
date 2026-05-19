#!/usr/bin/env python3
"""AAlgoI Demo — RL-powered algorithm selection across 3 domains."""

from pipeline import UniversalSolver
from core.problem_spec import ProblemSpec, ProblemType


def main():
    solver = UniversalSolver()

    # 1. Sorting
    print("=== SORTING DEMO ===")
    data = [64, 34, 25, 12, 22, 11, 90, 5]
    spec = ProblemSpec(name="sort_demo", problem_type=ProblemType.TRANSFORMATION)
    result = solver.solve(spec, data)
    print(f"  Sorted: {result['result']}")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Time: {result['time_ms']:.2f}ms")

    # 2. Pathfinding
    print("\n=== PATHFINDING DEMO ===")
    graph = {
        'A': {'B': 4, 'C': 2},
        'B': {'D': 5},
        'C': {'D': 1},
        'D': {}
    }
    spec = ProblemSpec(name="path_demo", problem_type=ProblemType.PATHFINDING)
    result = solver.solve(spec, {'graph': graph, 'start': 'A', 'end': 'D'})
    print(f"  Path: {result['result']}")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Time: {result['time_ms']:.2f}ms")

    # 3. Optimization
    print("\n=== OPTIMIZATION DEMO ===")
    items = [
        {'value': 60, 'weight': 10},
        {'value': 100, 'weight': 20},
        {'value': 120, 'weight': 30},
    ]
    spec = ProblemSpec(name="knapsack_demo", problem_type=ProblemType.OPTIMIZATION)
    result = solver.solve(spec, {'items': items, 'capacity': 50})
    out = result['result']
    if isinstance(out, dict):
        print(f"  Selected items: {out.get('selected', [])}")
        print(f"  Total value: {out.get('value', 0)}")
    else:
        print(f"  Result: {out}")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Time: {result['time_ms']:.2f}ms")


if __name__ == "__main__":
    main()
