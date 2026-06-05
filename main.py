
#!/usr/bin/env python3
"""
AAlgoI - Artificial Algorithm Intelligence
Main entry point for the adaptive algorithm system.

Usage:
    python main.py --mode demo
    python main.py --mode benchmark --size 10000
    python main.py --mode interactive
    python main.py --mode learning --llm-enabled --llm-model phi3:mini
"""

import argparse
import random
import time
import json
from typing import List, Dict, Any

from aalgoi.pipeline import AAlgoI

def generate_test_data(size: int, pattern: str = "random") -> List[int]:
    if pattern == "random":
        return [random.randint(0, size * 10) for _ in range(size)]
    elif pattern == "sorted":
        return list(range(size))
    elif pattern == "reverse":
        return list(range(size, 0, -1))
    elif pattern == "nearly_sorted":
        data = list(range(size))
        swaps = size // 20
        for _ in range(swaps):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            data[i], data[j] = data[j], data[i]
        return data
    elif pattern == "few_unique":
        return [random.randint(0, 10) for _ in range(size)]
    else:
        return [random.randint(0, size * 10) for _ in range(size)]

def demo_mode(config: Dict = None):
    print("=" * 70)
    print("  AAlgoI - Artificial Algorithm Intelligence v2.0")
    print("  Adaptive Algorithm Selection & Optimization Demo")
    print("=" * 70)
    print()

    system = AAlgoI(config={
        "strategy": "hybrid",
        "time_budget_ms": 500,
        "priority": "balanced",
        **(config or {})
    })

    test_cases = [
        ("tiny_random", 15, "random"),
        ("small_sorted", 100, "sorted"),
        ("medium_random", 5000, "random"),
        ("large_random", 50000, "random"),
        ("nearly_sorted", 10000, "nearly_sorted"),
        ("reverse_sorted", 5000, "reverse"),
    ]

    print("Running adaptive algorithm selection on various datasets...")
    print("-" * 70)

    for name, size, pattern in test_cases:
        data = generate_test_data(size, pattern)
        expected = sorted(data)

        print(f"\n  Test: {name} (size={size}, pattern={pattern})")

        result = system.run(data, task_type="sorting", expected_result=expected)

        stats = system.get_stats()
        pipeline = stats["active_pipeline"]

        if system.tracker.measurements:
            last_metric = system.tracker.measurements[-1]["metrics"]
            print(f"   Selected: {' -> '.join(pipeline)}")
            print(f"   Time: {last_metric['wall_time_ms']:.2f}ms | "
                  f"Quality: {last_metric['quality_score']:.2f} | "
                  f"Success: {last_metric['success']}")

        explanation = system.explain_decision()
        for reason in explanation.get("reasoning", [])[:3]:
            print(f"   {reason}")

    print()
    print("-" * 70)
    print("\n  System Statistics:")
    stats = system.get_stats()
    print(f"   Total executions: {stats['executions']}")
    print(f"   Average time: {stats['avg_time_ms']:.2f}ms")
    print(f"   Knowledge base records: {stats['knowledge_base']['total_records']}")
    print(f"   Meta-controller strategy: {stats['meta_controller']['strategy']}")
    print(f"   History size: {stats['meta_controller']['history_size']}")
    print(f"   Hot cache entries: {stats['hot_cache_size']}")
    print(f"   Drift events: {stats['drift']['drift_count']}")
    print(f"   Validation failures: {stats['validator']['total_failures']}")
    if stats['meta_controller']['llm']['enabled']:
        print(f"   LLM: {stats['meta_controller']['llm']['model']} ({'available' if stats['meta_controller']['llm']['available'] else 'unavailable'})")

    print()
    print("\n  Algorithm Performance Summary:")
    for algo_name, summary in stats["performance_tracker"]["summaries"].items():
        if summary:
            print(f"   {algo_name:15s}: avg={summary.get('avg_time_ms', 0):8.2f}ms, "
                  f"quality={summary.get('avg_quality', 0):.2f}, "
                  f"success_rate={summary.get('success_rate', 0):.1%}")

def benchmark_mode(size: int = 10000, config: Dict = None):
    print("=" * 70)
    print(f"  AAlgoI Benchmark - Size: {size}")
    print("=" * 70)
    print()

    system = AAlgoI(config=config or {})

    patterns = ["random", "sorted", "nearly_sorted", "reverse", "few_unique"]

    for pattern in patterns:
        print(f"\n  Pattern: {pattern}")
        data = generate_test_data(size, pattern)

        results = system.benchmark(data)

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["time_ms"] if x[1]["success"] else float('inf')
        )

        for algo_name, result in sorted_results:
            status = "+" if result["success"] and result["correct"] else "-"
            print(f"   {status} {algo_name:15s}: {result['time_ms']:8.2f}ms")

        result = system.run(data, task_type="sorting")
        chosen = system.get_stats()["active_pipeline"]
        print(f"   * AAlgoI chose: {' -> '.join(chosen)}")

def learning_demo(config: Dict = None):
    print("=" * 70)
    print("  AAlgoI Learning Demo v2.0")
    print("  Showing how the system improves over time with bandit + vector KB")
    print("=" * 70)
    print()

    system = AAlgoI(config={
        "strategy": "hybrid",
        "retrain_interval": 50,
        **(config or {})
    })

    print("Training phase: Running 200 executions with varying data...")

    for i in range(200):
        size = random.randint(100, 10000)
        pattern = random.choice(["random", "sorted", "nearly_sorted", "reverse"])
        data = generate_test_data(size, pattern)
        expected = sorted(data)

        system.run(data, task_type="sorting", expected_result=expected)

        if (i + 1) % 50 == 0:
            stats = system.get_stats()
            print(f"  Completed {i+1} executions... (confidence: {stats['meta_controller']['last_confidence']:.2f})")

    print()
    print("Testing phase: Evaluating on new data...")

    test_cases = [
        ("small_random", 50, "random"),
        ("medium_sorted", 5000, "sorted"),
        ("large_nearly_sorted", 20000, "nearly_sorted"),
    ]

    for name, size, pattern in test_cases:
        data = generate_test_data(size, pattern)
        expected = sorted(data)

        result = system.run(data, task_type="sorting", expected_result=expected)

        stats = system.get_stats()
        pipeline = stats["active_pipeline"]

        if system.tracker.measurements:
            last_metric = system.tracker.measurements[-1]["metrics"]
            print(f"\n{name}: {' -> '.join(pipeline)}")
            print(f"  Time: {last_metric['wall_time_ms']:.2f}ms, "
                  f"Quality: {last_metric['quality_score']:.2f}, "
                  f"Confidence: {stats['meta_controller']['last_confidence']:.0%}")

    print()
    print("\n  Final Knowledge Base Stats:")
    kb_stats = system.knowledge.get_all_stats()
    for algo, stats in kb_stats.items():
        print(f"  {algo}: {stats['total_executions']} runs, "
              f"avg_time={stats['avg_time_ms']:.2f}ms, "
              f"success={stats['success_rate']:.1%}, "
              f"score={stats.get('avg_score', 0):.2f}")

    print()
    print("  Bandit Stats:")
    bandit_stats = system.meta_controller.bandit.get_stats()
    print(f"  Epsilon: {bandit_stats['epsilon']:.3f}")
    print(f"  Total trials: {bandit_stats['total_trials']}")
    for name, bstats in bandit_stats['algorithm_stats'].items():
        if bstats['count'] > 0:
            print(f"  {name:15s}: n={bstats['count']:4d}, avg_reward={bstats['avg_reward']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="AAlgoI - Artificial Algorithm Intelligence v2.0")
    parser.add_argument("--mode", choices=["demo", "benchmark", "interactive", "learning"],
                       default="demo", help="Execution mode")
    parser.add_argument("--size", type=int, default=10000, help="Data size for benchmark")
    parser.add_argument("--strategy", choices=["rule-based", "ml-based", "genetic", "hybrid"],
                       default="hybrid", help="Meta-controller strategy")
    parser.add_argument("--llm-enabled", action="store_true", help="Enable LLM advisor")
    parser.add_argument("--llm-backend", choices=["ollama", "llamacpp"], default="ollama",
                       help="LLM backend")
    parser.add_argument("--llm-model", default="phi3:mini", help="LLM model name")
    parser.add_argument("--priority", choices=["speed", "accuracy", "balanced"],
                       default="balanced", help="Execution priority")
    parser.add_argument("--time-budget", type=int, default=500, help="Time budget in ms")
    parser.add_argument("--no-dag", action="store_true", help="Disable DAG pipeline")

    args = parser.parse_args()

    config = {}
    config["strategy"] = args.strategy
    config["priority"] = args.priority
    config["time_budget_ms"] = args.time_budget
    config["enable_dag"] = not args.no_dag
    config["llm"] = {
        "enabled": args.llm_enabled,
        "backend": args.llm_backend,
        "model": args.llm_model
    }

    if args.mode == "demo":
        demo_mode(config)
    elif args.mode == "benchmark":
        benchmark_mode(args.size, config)
    elif args.mode == "interactive":
        print("Interactive mode not available in this environment")
    elif args.mode == "learning":
        learning_demo(config)

if __name__ == "__main__":
    main()
