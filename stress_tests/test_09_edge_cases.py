import random
from aalgoi.core.smart_solver import SmartSolver

EDGE_CASES = [
    ("sort empty list",         [],                    True),
    ("find path",               {"graph": {}, "start": "A", "end": "B"}, False),
    ("sort this",               [42],                  True),
    ("sort this",               ["a"],                 True),
    ("sort this",               [7]*10000,             True),
    ("sort this",               list(range(10000)),    True),
    ("sort this",               list(range(10000, 0, -1)), True),
    ("sort this",               [float('inf'), float('-inf'), 0], True),
    ("sort this",               [10**18, -10**18, 0],  True),
    ("sort this",               [1, "two", 3.0, None], False),
    ("sort this",               [random.randint(0, 10**9) for _ in range(1_000_000)], True),
    ("find shortest path",
     {
         "graph": {f"n{i}": {f"n{i+1}": 1} for i in range(10000)},
         "start": "n0",
         "end": "n9999"
     }, True),
    ("sort these words",
     ["hello", "world", "apple", "zebra", "banana"], True),
    ("sort this",
     [random.randint(-10**9, 10**9) for _ in range(10000)], True),
    ("sort this",
     [1.0, float('nan'), 3.0, float('nan'), 2.0], False),
    ("sort this",               None,                  False),
    ("knapsack problem",
     {"items": [{"weight": 1, "value": 10}], "capacity": 0}, True),
    ("find shortest path",
     {"graph": {"A": {}, "B": {}}, "start": "A", "end": "B"}, False),
]

def test_extreme_edge_cases():
    solver = SmartSolver()

    crashes    = []
    wrong_fail = []

    for i, (description, data, should_succeed) in enumerate(EDGE_CASES):
        try:
            result = solver.ask(description, data)
            succeeded = result.get('success', False)

            if should_succeed and not succeeded:
                wrong_fail.append(
                    f"Case {i}: '{description}' with "
                    f"data={str(data)[:50]} - "
                    f"should succeed but failed: {result.get('error', 'unknown')}"
                )
        except Exception as e:
            crashes.append(
                f"Case {i}: '{description}' CRASHED: {type(e).__name__}: {e}"
            )

    print(f"\nEdge Case Results:")
    print(f"  Total cases: {len(EDGE_CASES)}")
    print(f"  Crashes:     {len(crashes)}")
    print(f"  Wrong fails: {len(wrong_fail)}")

    for c in crashes:
        print(f"  CRASH: {c}")
    for w in wrong_fail:
        print(f"  WRONG FAIL: {w}")

    assert len(crashes) == 0, \
        f"FAIL: System crashed on {len(crashes)} edge cases"
    assert len(wrong_fail) == 0, \
        f"FAIL: System failed on {len(wrong_fail)} cases that should succeed"
