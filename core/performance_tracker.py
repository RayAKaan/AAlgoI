
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

class PerformanceTracker:
    """
    Advanced performance tracking with multi-dimensional metrics.
    Tracks time, memory, accuracy, and custom quality scores.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.measurements: List[Dict] = []
        self._baseline_measurements: Dict[str, List[float]] = {}

    def evaluate(self, fn: Callable, data: Any, context: Dict, 
                 expected_result: Any = None) -> Tuple[Any, Dict]:
        """
        Execute function and measure comprehensive performance metrics.
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Pre-execution metrics
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_before = process.cpu_percent()

        # Execute with timing
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        try:
            result = fn(data)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            traceback_str = traceback.format_exc()

        end_time = time.perf_counter()
        end_cpu = time.process_time()

        # Post-execution metrics
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_after = process.cpu_percent()

        # Calculate metrics
        wall_time_ms = (end_time - start_time) * 1000
        cpu_time_ms = (end_cpu - start_cpu) * 1000
        memory_delta_mb = mem_after - mem_before

        # Quality metrics
        quality_score = self._compute_quality_score(
            result, expected_result, context, success
        )

        # Budget compliance
        time_budget = context.get("constraints", {}).get("time_budget_ms", float('inf'))
        within_budget = wall_time_ms <= time_budget

        metrics = {
            "wall_time_ms": wall_time_ms,
            "cpu_time_ms": cpu_time_ms,
            "memory_delta_mb": memory_delta_mb,
            "memory_peak_mb": max(mem_before, mem_after),
            "within_budget": within_budget,
            "budget_utilization": wall_time_ms / time_budget if time_budget != float('inf') else 0,
            "quality_score": quality_score,
            "success": success,
            "error": error,
            "throughput": len(data) / (wall_time_ms / 1000) if wall_time_ms > 0 and hasattr(data, '__len__') else 0,
        }

        # Record measurement
        measurement = {
            "timestamp": time.time(),
            "context_features": context.get("features", {}),
            "metrics": metrics,
            "algorithm_names": context.get("algorithms", []),
        }
        self.measurements.append(measurement)

        # Update baselines
        for algo_name in context.get("algorithms", []):
            if algo_name not in self._baseline_measurements:
                self._baseline_measurements[algo_name] = []
            self._baseline_measurements[algo_name].append(wall_time_ms)

        return result, metrics

    def _compute_quality_score(self, result: Any, expected: Any, 
                               context: Dict, success: bool) -> float:
        """
        Compute quality score from 0 to 1.
        """
        if not success:
            return 0.0

        if expected is not None:
            # Compare with expected result
            try:
                if isinstance(result, list) and isinstance(expected, list):
                    if len(result) != len(expected):
                        return 0.5  # Partial credit

                    # Check if sorted correctly
                    if result == sorted(expected):
                        correct_count = sum(1 for a, b in zip(result, expected) if a == b)
                        return correct_count / len(expected)
                    else:
                        return 0.0  # Not sorted
                else:
                    return 1.0 if result == expected else 0.0
            except:
                return 0.5

        # No expected result - use heuristics
        if isinstance(result, list):
            # Check if result is sorted
            is_sorted = all(result[i] <= result[i+1] for i in range(len(result)-1))
            return 1.0 if is_sorted else 0.0

        return 1.0 if result is not None else 0.0

    def get_performance_summary(self, algorithm_name: Optional[str] = None) -> Dict:
        """
        Get performance summary for an algorithm or overall.
        """
        if algorithm_name:
            measurements = [
                m for m in self.measurements 
                if algorithm_name in m.get("algorithm_names", [])
            ]
        else:
            measurements = self.measurements

        if not measurements:
            return {}

        times = [m["metrics"]["wall_time_ms"] for m in measurements]
        qualities = [m["metrics"]["quality_score"] for m in measurements]

        return {
            "count": len(measurements),
            "avg_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "avg_quality": np.mean(qualities),
            "success_rate": sum(1 for m in measurements if m["metrics"]["success"]) / len(measurements),
            "budget_compliance": sum(1 for m in measurements if m["metrics"]["within_budget"]) / len(measurements)
        }

    def get_baseline(self, algorithm_name: str) -> Optional[float]:
        """Get average baseline performance for an algorithm."""
        measurements = self._baseline_measurements.get(algorithm_name, [])
        if measurements:
            return np.median(measurements)
        return None
