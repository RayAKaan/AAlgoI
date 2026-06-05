import random

from aalgoi.core.smart_solver import SmartSolver

DOMAIN_TEST_CASES = [
    {
        "description": "sort these numbers",
        "data": [random.randint(-1000, 1000) for _ in range(100)],
        "forbidden_algorithms": ["dijkstra", "a_star", "bfs_path",
                                  "kmeans", "pca_reduction", "greedy_knapsack"]
    },
    {
        "description": "order this list from smallest to largest",
        "data": [3.14, 2.71, 1.41, 1.73, 0.57],
        "forbidden_algorithms": ["dijkstra", "a_star", "bfs_path"]
    },
    {
        "description": "arrange these words alphabetically",
        "data": ["banana", "apple", "cherry", "date", "elderberry"],
        "forbidden_algorithms": ["dijkstra", "a_star", "bfs_path"]
    },
    {
        "description": "sort descending",
        "data": list(range(1000, 0, -1)),
        "forbidden_algorithms": ["dijkstra", "a_star", "bfs_path"]
    },
    {
        "description": "sort this nearly sorted list",
        "data": list(range(10000)) + [random.randint(0,10000) for _ in range(10)],
        "forbidden_algorithms": ["identity"]
    },
    {
        "description": "find shortest path from A to B",
        "data": {
            "graph": {
                "A": {"B": 1, "C": 4},
                "B": {"C": 2, "D": 5},
                "C": {"D": 1},
                "D": {}
            },
            "start": "A",
            "end": "D"
        },
        "forbidden_algorithms": ["quicksort", "mergesort",
                                  "timsort", "kmeans"]
    },
    {
        "description": "navigate from start to goal",
        "data": {
            "graph": {f"node_{i}": {
                f"node_{j}": random.randint(1, 10)
                for j in random.sample(range(50), 3)
                if j != i
            } for i in range(50)},
            "start": "node_0",
            "end": "node_49"
        },
        "forbidden_algorithms": ["quicksort", "mergesort", "timsort"]
    },
    {
        "description": "knapsack problem maximize value within weight limit",
        "data": {
            "items": [
                {"weight": random.randint(1, 20),
                 "value": random.randint(1, 100)}
                for _ in range(50)
            ],
            "capacity": 100
        },
        "forbidden_algorithms": ["quicksort", "mergesort",
                                  "dijkstra", "a_star"]
    },
    {
        "description": "find the path through these sorted numbers",
        "data": [1, 3, 5, 7, 9, 11],
        "forbidden_algorithms": ["dijkstra", "a_star", "bfs_path"]
    },
    {
        "description": "optimize the order of this list",
        "data": [9, 4, 7, 1, 3, 8, 2, 6, 5],
        "forbidden_algorithms": ["dijkstra", "a_star"]
    },
]

def test_domain_routing_never_misroutes():
    solver = SmartSolver()
    failures = []

    for i, case in enumerate(DOMAIN_TEST_CASES):
        result = solver.ask(case['description'], case['data'])
        chosen_algo = result.get('algorithm', '')
        chosen_algo_lower = chosen_algo.lower()

        for forbidden in case.get('forbidden_algorithms', []):
            if forbidden.lower() in chosen_algo_lower:
                failures.append(
                    f"Case {i}: Forbidden algorithm '{forbidden}' was chosen "
                    f"(got '{chosen_algo}') | '{case['description']}'"
                )

    print("\nDomain Routing Results:")
    print(f"  Total cases: {len(DOMAIN_TEST_CASES)}")
    print(f"  Failures:    {len(failures)}")
    for f in failures:
        print(f"    FAIL: {f}")

    assert len(failures) == 0, \
        f"Domain routing failed on {len(failures)} cases"
