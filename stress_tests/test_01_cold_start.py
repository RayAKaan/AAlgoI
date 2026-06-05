import time
import threading
import statistics
from aalgoi.core.rl.agents.selection_agent import PPOAgent
from aalgoi.core.smart_solver import SmartSolver

def test_cold_start_under_concurrent_load():
    PPOAgent._clear_cache()

    results = []
    errors  = []
    times   = []

    def worker(thread_id):
        try:
            solver = SmartSolver()
            start  = time.perf_counter()

            result = solver.ask(
                "sort this list",
                [5, 3, 1, 4, 2, 8, 6, 7, 9, 0]
            )

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            assert result.get('success') == True
            assert result.get('result') == [0,1,2,3,4,5,6,7,8,9], \
                f"Thread {thread_id} got wrong result: {result.get('result')}"

            results.append(result)

        except Exception as e:
            errors.append(f"Thread {thread_id}: {str(e)}")

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(20)
    ]

    start_all = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - start_all

    print(f"\nCold Start Under Load Results:")
    print(f"  Threads:       20")
    print(f"  Successes:     {len(results)}")
    print(f"  Errors:        {len(errors)}")
    print(f"  Total time:    {total_time:.3f}s")
    print(f"  Median latency: {statistics.median(times)*1000:.1f}ms" if times else "  N/A")
    if len(times) >= 19:
        print(f"  P95 latency:   {sorted(times)[18]*1000:.1f}ms")
    if len(times) >= 20:
        print(f"  P99 latency:   {sorted(times)[19]*1000:.1f}ms")

    if errors:
        print(f"\n  FAILURES:")
        for e in errors:
            print(f"    {e}")

    assert len(errors) == 0,   f"FAIL: Some threads crashed on cold start: {errors}"
    assert len(results) == 20, f"FAIL: Not all threads returned results"
    assert total_time < 30,    f"FAIL: Cold start took more than 30 seconds ({total_time:.1f}s)"
    assert statistics.median(times) < 4.0, \
        f"FAIL: Median cold start latency > 4 seconds ({statistics.median(times)*1000:.1f}ms)"
