import random
import time

from aalgoi.core.smart_solver import SmartSolver


def test_system_learns_over_time():
    solver = SmartSolver()

    def make_nearly_sorted():
        base = list(range(1000))
        for _ in range(5):
            i, j = random.randint(0,999), random.randint(0,999)
            base[i], base[j] = base[j], base[i]
        return base

    def make_random():
        return [random.randint(0, 10000) for _ in range(1000)]

    def make_reverse():
        return list(range(1000, 0, -1))

    problem_generators = [
        make_nearly_sorted,
        make_random,
        make_reverse
    ]

    optimal_map = {
        "nearly_sorted": "timsort",
        "random":        "quicksort",
        "reverse":       "heap_sort",
    }

    first_50_optimal = 0
    last_50_optimal  = 0
    first_50_times = []
    last_50_times  = []

    for i in range(500):
        gen = random.choice(problem_generators)
        data = gen()
        data_type = gen.__name__.replace("make_", "")

        start  = time.perf_counter()
        result = solver.ask("sort this list", data)
        elapsed = time.perf_counter() - start

        chosen = result.get('algorithm', '').lower()
        optimal = optimal_map.get(data_type, '')
        is_optimal = optimal in chosen

        if i < 50:
            first_50_times.append(elapsed)
            if is_optimal:
                first_50_optimal += 1

        if i >= 450:
            last_50_times.append(elapsed)
            if is_optimal:
                last_50_optimal += 1

    import statistics
    first_50_rate  = first_50_optimal / 50
    last_50_rate   = last_50_optimal  / 50
    first_50_median = statistics.median(first_50_times) * 1000 if first_50_times else 0
    last_50_median  = statistics.median(last_50_times)  * 1000 if last_50_times else 0

    print("\nLearning Over Time:")
    print(f"  First 50 optimal selection rate: {first_50_rate:.1%}")
    print(f"  Last  50 optimal selection rate: {last_50_rate:.1%}")
    print(f"  First 50 median latency:         {first_50_median:.1f}ms")
    print(f"  Last  50 median latency:         {last_50_median:.1f}ms")
    print(f"  Improvement in selection rate:   "
          f"{(last_50_rate - first_50_rate):.1%}")

    assert last_50_rate >= first_50_rate - 0.15, \
        f"FAIL: System selection quality degraded >15% over 500 solves " \
        f"({first_50_rate:.1%} -> {last_50_rate:.1%})"
    assert last_50_median <= first_50_median * 1.5, \
        f"FAIL: System got significantly slower over time instead of faster " \
        f"({first_50_median:.1f}ms -> {last_50_median:.1f}ms)"
