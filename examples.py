
#!/usr/bin/env python3
"""
AAlgoI - Example Usage Script v2.0

Demonstrates all new features: vector KB, validation, bandit, drift detection,
DAG pipelines, LLM advisor, decision audit log, and genetic evolution.
"""

from aalgoi.pipeline import AAlgoI
import random
import time
import numpy as np


def example_1_basic_sorting():
    """Basic sorting with automatic algorithm selection"""
    print("=" * 70)
    print("Example 1: Basic Sorting")
    print("=" * 70)

    system = AAlgoI(config={
        "strategy": "hybrid",
        "time_budget_ms": 500
    })

    datasets = [
        ("Tiny random", [random.randint(0, 100) for _ in range(10)]),
        ("Small sorted", list(range(100))),
        ("Medium random", [random.randint(0, 10000) for _ in range(5000)]),
        ("Nearly sorted", _make_nearly_sorted(10000)),
        ("Large random", [random.randint(0, 100000) for _ in range(50000)]),
    ]

    for name, data in datasets:
        expected = sorted(data)
        result = system.run(data, task_type="sorting", expected_result=expected)

        pipeline = system.get_stats()["active_pipeline"]
        last_metric = system.tracker.measurements[-1]["metrics"]

        print(f"\n{name}:")
        print(f"  Selected: {' -> '.join(pipeline)}")
        print(f"  Time: {last_metric['wall_time_ms']:.2f}ms")
        print(f"  Quality: {last_metric['quality_score']:.2f}")
        print(f"  Correct: {result == expected}")


def example_2_strategy_comparison():
    """Compare different selection strategies"""
    print("\n" + "=" * 70)
    print("Example 2: Strategy Comparison (with bandit + confidence)")
    print("=" * 70)

    strategies = ["rule-based", "hybrid"]
    data = [random.randint(0, 10000) for _ in range(10000)]
    expected = sorted(data)

    for strategy in strategies:
        system = AAlgoI(config={"strategy": strategy})

        start = time.perf_counter()
        result = system.run(data, task_type="sorting", expected_result=expected)
        elapsed = (time.perf_counter() - start) * 1000

        stats = system.get_stats()
        mc_stats = stats["meta_controller"]
        pipeline = stats["active_pipeline"]
        print(f"\n{strategy:15s}: {' -> '.join(pipeline):20s} | "
              f"{elapsed:8.2f}ms | confidence={mc_stats['last_confidence']:.0%}")


def example_3_learning_over_time():
    """Show how system learns and improves with bandit exploration"""
    print("\n" + "=" * 70)
    print("Example 3: Learning Over Time (with bandit evolution)")
    print("=" * 70)

    system = AAlgoI(config={
        "strategy": "hybrid",
        "retrain_interval": 50
    })

    print("\nTraining phase (100 executions)...")
    for i in range(100):
        size = random.randint(100, 5000)
        pattern = random.choice(["random", "sorted", "nearly_sorted"])

        if pattern == "random":
            data = [random.randint(0, size * 10) for _ in range(size)]
        elif pattern == "sorted":
            data = list(range(size))
        else:
            data = _make_nearly_sorted(size)

        expected = sorted(data)
        system.run(data, task_type="sorting", expected_result=expected)

    stats = system.get_stats()
    mc_stats = stats["meta_controller"]
    print(f"History size: {mc_stats['history_size']}")
    print(f"Knowledge records: {stats['knowledge_base']['total_records']}")
    print(f"Bandit epsilon: {mc_stats['bandit']['epsilon']:.3f}")
    print(f"Last confidence: {mc_stats['last_confidence']:.0%}")

    print("\nAlgorithm Performance:")
    for algo_name, summary in stats["performance_tracker"]["summaries"].items():
        if summary and summary.get("count", 0) > 0:
            print(f"  {algo_name:15s}: avg={summary['avg_time_ms']:8.2f}ms, n={summary['count']}")


def example_4_explain_decisions():
    """Get explanations for algorithm selection with audit trail"""
    print("\n" + "=" * 70)
    print("Example 4: Decision Explanation + Audit Trail")
    print("=" * 70)

    system = AAlgoI()

    data = _make_nearly_sorted(5000)
    expected = sorted(data)
    result = system.run(data, task_type="sorting", expected_result=expected)

    explanation = system.explain_decision()
    print("\nContext:")
    for key, value in explanation["context"].items():
        print(f"  {key}: {value}")
    print(f"\nSelected: {' -> '.join(explanation['selected_algorithms'])}")
    print(f"Confidence: {explanation.get('confidence', 0):.0%}")
    print(f"Selection reason: {explanation.get('selection_reason', 'N/A')}")
    print("\nReasoning:")
    for reason in explanation["reasoning"]:
        print(f"  - {reason}")

    print("\nDecision Log:")
    log = system.get_decision_log(5)
    for entry in log:
        print(f"  [{entry['timestamp']:.0f}] {entry['chosen']:15s} | "
              f"conf={entry['confidence']:.0%} | ok={entry['success']}")


def example_5_benchmark():
    """Benchmark all algorithms"""
    print("\n" + "=" * 70)
    print("Example 5: Algorithm Benchmark")
    print("=" * 70)

    system = AAlgoI()
    data = [random.randint(0, 100000) for _ in range(10000)]

    print("\nBenchmarking all sorting algorithms...")
    results = system.benchmark(data)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["time_ms"] if x[1]["success"] else float("inf")
    )

    for algo_name, result in sorted_results:
        status = "+" if result["success"] and result.get("correct", False) else "-"
        print(f"  {status} {algo_name:15s}: {result['time_ms']:8.2f}ms")

    result = system.run(data, task_type="sorting")
    chosen = system.get_stats()["active_pipeline"]
    print(f"\n* AAlgoI chose: {' -> '.join(chosen)}")


def example_6_validation_and_drift():
    """Demonstrate output validation and drift detection"""
    print("\n" + "=" * 70)
    print("Example 6: Output Validation + Drift Detection")
    print("=" * 70)

    system = AAlgoI()

    print("\nRunning 50 normal executions...")
    for i in range(50):
        data = [random.randint(0, 1000) for _ in range(100)]
        expected = sorted(data)
        system.run(data, task_type="sorting", expected_result=expected)

    print(f"Validation stats: {system.get_validation_stats()}")
    print(f"Drift stats: {system.get_drift_stats()}")


def example_7_llm_advisor():
    """LLM-powered decision explanation (requires ollama or llama.cpp)"""
    print("\n" + "=" * 70)
    print("Example 7: LLM Advisor (requires ollama/llama.cpp)")
    print("=" * 70)

    system = AAlgoI(config={
        "llm": {
            "enabled": True,
            "backend": "ollama",
            "model": "phi3:mini"
        }
    })

    data = _make_nearly_sorted(5000)
    expected = sorted(data)
    result = system.run(data, task_type="sorting", expected_result=expected)

    if system.meta_controller.llm.check_available():
        explanation = system.explain_decision()
        print(f"\nLLM Explanation: {explanation.get('llm_explanation', 'N/A')}")
    else:
        print("\nLLM not available. Start ollama: 'ollama pull phi3:mini' then 'ollama serve'")


def example_8_dag_pipeline():
    """DAG pipeline execution (parallel branches)"""
    print("\n" + "=" * 70)
    print("Example 8: DAG Pipeline (Parallel Image Processing)")
    print("=" * 70)

    try:
        import numpy as np
        system = AAlgoI(config={"enable_dag": True})

        image = np.random.rand(100, 100).astype(np.float32)

        result = system.run(image, task_type="image_processing")
        pipeline = system.get_stats()["active_pipeline"]
        print(f"\nPipeline: {' -> '.join(pipeline)}")
        print(f"DAG mode: DAG pipeline executed")

    except ImportError:
        print("\nSkipping - numpy required for image processing demo")


def _make_nearly_sorted(size):
    data = list(range(size))
    swaps = size // 20
    for _ in range(swaps):
        i, j = random.randint(0, size-1), random.randint(0, size-1)
        data[i], data[j] = data[j], data[i]
    return data


if __name__ == "__main__":
    example_1_basic_sorting()
    example_2_strategy_comparison()
    example_3_learning_over_time()
    example_4_explain_decisions()
    example_5_benchmark()
    example_6_validation_and_drift()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
