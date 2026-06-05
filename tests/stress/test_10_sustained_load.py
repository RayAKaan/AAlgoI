import random
import time

from aalgoi.core.smart_solver import SmartSolver


def test_sustained_load_no_degradation():
    solver = SmartSolver()

    window_size = 100
    results_window = []
    latency_windows = {'first': [], 'last': []}
    total_crashes = 0
    total_solves = 0
    start_time = time.time()
    max_duration = 300
    max_solves = 5000

    problem_pool = [
        lambda: ("sort this", [random.randint(0, 10**6)
                                for _ in range(random.randint(10, 10000))]),
        lambda: ("sort nearly sorted",
                 list(range(10000)) + [random.randint(0,10000) for _ in range(10)]),
        lambda: ("find shortest path", {
            "graph": {
                f"n{i}": {
                    f"n{random.randint(0,49)}": random.randint(1,10)
                    for _ in range(3)
                }
                for i in range(50)
            },
            "start": "n0", "end": "n49"
        }),
        lambda: ("knapsack", {
            "items": [
                {"weight": random.randint(1,20),
                 "value": random.randint(1,100)}
                for _ in range(random.randint(10, 100))
            ],
            "capacity": random.randint(50, 200)
        }),
    ]

    print(f"\nSustained Load Test — running for up to "
          f"{max_duration}s or {max_solves} solves")
    print(f"{'Solve':>8} {'Success%':>10} {'#Samples':>10}")

    while (time.time() - start_time < max_duration and total_solves < max_solves):
        try:
            desc, data = random.choice(problem_pool)()

            t0 = time.perf_counter()
            result = solver.ask(desc, data)
            elapsed = (time.perf_counter() - t0) * 1000

            results_window.append(result.get('success', False))

            if total_solves < 500:
                latency_windows['first'].append(elapsed)
            if total_solves >= max_solves - 500:
                latency_windows['last'].append(elapsed)

            if len(results_window) > window_size:
                results_window.pop(0)

            total_solves += 1

            if total_solves % 100 == 0:
                success_rate = sum(results_window) / len(results_window)

                print(f"{total_solves:>8} {success_rate:>10.1%} "
                      f"{len(results_window):>10}")

                if success_rate < 0.95 and total_solves > 200:
                    print(f"\nFAIL: Success rate dropped to "
                          f"{success_rate:.1%} at solve {total_solves}")
                    break

        except Exception as e:
            total_crashes += 1
            print(f"  CRASH at solve {total_solves}: {e}")
            if total_crashes > 10:
                print("  Too many crashes. Stopping.")
                break

    import statistics
    first_p50 = statistics.median(latency_windows['first']) if latency_windows['first'] else 0
    last_p50  = statistics.median(latency_windows['last'])  if latency_windows['last']  else 0

    final_success_rate = sum(results_window) / len(results_window) if results_window else 0

    print(f"\n{'='*50}")
    print("Sustained Load Final Results:")
    print(f"  Total solves:       {total_solves}")
    print(f"  Total crashes:      {total_crashes}")
    print(f"  Final success rate: {final_success_rate:.1%}")
    print(f"  Duration:           {(time.time()-start_time)/60:.1f} minutes")

    assert total_crashes == 0, \
        f"FAIL: {total_crashes} crashes during sustained load"
    assert final_success_rate >= 0.95, \
        f"FAIL: Final success rate {final_success_rate:.1%} < 95%"
    assert last_p50 <= first_p50 * 2, \
        f"FAIL: Latency doubled. {first_p50:.1f}ms -> {last_p50:.1f}ms"

    if total_solves < 100:
        print("  WARNING: Too few solves to draw conclusions")
