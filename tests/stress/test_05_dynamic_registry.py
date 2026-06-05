import time

from aalgoi.algorithms.base import Algorithm
from aalgoi.core.smart_solver import SmartSolver


class PerfectSorter(Algorithm):
    name = "PerfectSorter"
    metadata = {
        "complexity": {
            "best": "O(n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)"
        },
        "properties": {
            "is_stable": True,
            "is_in_place": False,
            "is_deterministic": True,
            "best_on_nearly_sorted": True,
            "best_on_large_data": True,
        },
        "problem_types": ["SORTING"]
    }

    def process(self, data):
        return sorted(data)

    def validate_output(self, input_data, output):
        return output == sorted(input_data)

def test_new_algorithm_immediately_selectable():
    solver = SmartSolver()
    usolver = solver.solver

    pre_algos_used = []
    for _ in range(10):
        result = solver.ask("sort this list", [5,3,1,4,2,8,6,7])
        pre_algos_used.append(result.get('algorithm', ''))

    assert "PerfectSorter" not in [a for a in pre_algos_used], \
        "PerfectSorter should not exist yet"

    registration_time = time.perf_counter()
    usolver.registry["PerfectSorter"] = PerfectSorter()
    registration_latency = (time.perf_counter() - registration_time) * 1000

    post_algos_used = []
    for _ in range(10):
        result = solver.ask("sort this nearly sorted list",
                            list(range(1000)) + [999, 998, 997])
        post_algos_used.append(result.get('algorithm', ''))

    print("\nDynamic Registry Test:")
    print(f"  Registration latency: {registration_latency:.1f}ms")
    print(f"  Algos used pre-registration: {set(pre_algos_used)}")
    print(f"  Algos used post-registration: {set(post_algos_used)}")
    print(f"  PerfectSorter selected: "
          f"{'Yes' if 'PerfectSorter' in post_algos_used else 'No'}")

    assert registration_latency < 100, \
        f"FAIL: Registration took {registration_latency:.1f}ms, max is 100ms"
