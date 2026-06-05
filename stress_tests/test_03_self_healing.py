from aalgoi.algorithms.base import Algorithm
from aalgoi.core.smart_solver import SmartSolver


class AlwaysCrashesAlgorithm(Algorithm):
    name = "AlwaysCrashes"
    def process(self, data):
        raise RuntimeError("I always crash. No exceptions. Pun intended.")

def test_self_healing_under_sabotage():
    solver = SmartSolver()
    usolver = solver.solver
    usolver.registry["AlwaysCrashes"] = AlwaysCrashesAlgorithm()

    data = list(range(100, 0, -1))
    expected = list(range(1, 101))

    successes = 0
    correct_results = 0

    for trial in range(20):
        result = solver.ask("sort this list", data)
        if result.get('success'):
            successes += 1
            if result.get('result') == expected:
                correct_results += 1

    print("\nSelf-Healing Under Sabotage:")
    print("  Trials:           20")
    print(f"  Successes:        {successes}/20")
    print(f"  Correct results:  {correct_results}/20")

    assert successes == 20, \
        f"FAIL: System should always recover. Got {successes}/20"
    assert correct_results == 20, \
        f"FAIL: All results should be correct. Got {correct_results}/20"
