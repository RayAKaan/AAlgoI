"""
Stress test: cross-domain knowledge graph routing.
"""
import random
from aalgoi.core.smart_solver import SmartSolver


def test_domain_native_routing_unchanged():
    """Sorting with sufficient candidates should still use sorting algos."""
    solver = SmartSolver()
    result = solver.ask("sort this list", [random.randint(0, 1000) for _ in range(100)])
    algo = result.get("algorithm", "")
    print(f"\nSorting (100 random): {algo}")
    assert algo in ("timsort", "quicksort", "insertion_sort", "merge_sort",
                    "heap_sort", "radix_sort"), f"Unexpected algo: {algo}"
    assert result.get("success", False), f"Sorting failed with {algo}"


def test_cross_domain_fallback_when_primary_fails():
    """Pathfinding with weighted graph selects dijkstra or cross-domain."""
    solver = SmartSolver()
    graph = {
        "graph": {"A": {"B": 5, "C": 2},
                   "B": {"D": 1},
                   "C": {"B": 1, "D": 4},
                   "D": {}},
        "start": "A",
        "end": "D",
    }
    result = solver.ask("find shortest path in weighted graph", graph)
    algo = result.get("algorithm", "")
    print(f"\nPathfinding (weighted): {algo} success={result.get('success')}")
    assert result.get("success", False), f"Pathfinding failed with {algo}"
    assert algo in ("dijkstra", "a_star", "bfs_path", "greedy_knapsack"), \
        f"Expected pathfinding or cross-domain algo, got: {algo}"
