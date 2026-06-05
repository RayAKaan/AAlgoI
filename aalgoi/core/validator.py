
from typing import Any, List, Dict, Optional, Callable, Tuple
from collections import defaultdict
import time
import numpy as np

from aalgoi.algorithms.base import Algorithm


class ValidationDetail:
    def __init__(self, metric: str, value: float, threshold: float,
                 passed: bool, message: str = ""):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.passed = passed
        self.message = message

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "message": self.message
        }


class LearningValidator:
    def __init__(self, adaptation_rate: float = 0.05):
        self.adaptation_rate = adaptation_rate
        self._thresholds: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "max_time_ms": 1000.0,
                "min_accuracy": 0.0,
                "max_memory_mb": 500.0,
                "output_not_null": 1.0
            }
        )
        self._history: Dict[str, List[Dict]] = defaultdict(list)
        self._adaptation_counts: Dict[str, int] = defaultdict(int)

    def validate(self, algorithm_name: str, input_data: Any,
                 output_data: Any, execution_time_ms: float = 0.0,
                 metadata: Optional[Dict] = None) -> Tuple[bool, List[ValidationDetail]]:
        details = []
        thresholds = self._thresholds[algorithm_name]

        not_null = output_data is not None
        details.append(ValidationDetail(
            metric="output_not_null", value=1.0 if not_null else 0.0,
            threshold=thresholds["output_not_null"], passed=not_null,
            message="" if not_null else f"{algorithm_name} returned None"
        ))

        time_ok = execution_time_ms <= thresholds["max_time_ms"]
        details.append(ValidationDetail(
            metric="max_time_ms", value=execution_time_ms,
            threshold=thresholds["max_time_ms"], passed=time_ok,
            message="" if time_ok else f"{execution_time_ms:.1f}ms > {thresholds['max_time_ms']:.1f}ms"
        ))

        if metadata and "accuracy" in metadata:
            acc = metadata["accuracy"]
            acc_ok = acc >= thresholds["min_accuracy"]
            details.append(ValidationDetail(
                metric="min_accuracy", value=acc,
                threshold=thresholds["min_accuracy"], passed=acc_ok,
                message="" if acc_ok else f"accuracy {acc:.2f} < {thresholds['min_accuracy']:.2f}"
            ))

        if metadata and "memory_mb" in metadata:
            mem = metadata["memory_mb"]
            mem_ok = mem <= thresholds["max_memory_mb"]
            details.append(ValidationDetail(
                metric="max_memory_mb", value=mem,
                threshold=thresholds["max_memory_mb"], passed=mem_ok,
                message="" if mem_ok else f"memory {mem:.1f}MB > {thresholds['max_memory_mb']:.1f}MB"
            ))

        passed = all(d.passed for d in details)
        return passed, details

    def record_result(self, algorithm_name: str, passed: bool,
                      details: List[ValidationDetail],
                      input_data: Any = None):
        record = {
            "passed": passed,
            "timestamp": time.time(),
            "details": [d.to_dict() for d in details],
            "input_size": len(input_data) if hasattr(input_data, '__len__') else None,
        }
        self._history[algorithm_name].append(record)

        if passed:
            self._adapt_thresholds(algorithm_name, details, input_data)

    def _adapt_thresholds(self, algorithm_name: str,
                          details: List[ValidationDetail],
                          input_data: Any = None):
        thresholds = self._thresholds[algorithm_name]
        self._adaptation_counts[algorithm_name] += 1
        count = self._adaptation_counts[algorithm_name]
        rate = self.adaptation_rate / max(1, count / 10)

        current_size = len(input_data) if hasattr(input_data, '__len__') else 1

        for d in details:
            if d.metric == "max_time_ms" and d.passed:
                ref_size = thresholds.get("_ref_size", current_size)
                size_factor = max(0.5, current_size / max(1, ref_size))
                scaled_threshold = d.threshold * size_factor
                margin = scaled_threshold - d.value

                if margin > scaled_threshold * 0.5 and count >= 5:
                    thresholds["max_time_ms"] = max(
                        d.value * 1.5,
                        thresholds["max_time_ms"] * (1 - rate)
                    )
                    thresholds["_ref_size"] = current_size

    def get_thresholds(self, algorithm_name: str) -> Dict[str, float]:
        return dict(self._thresholds[algorithm_name])

    def get_stats(self) -> Dict[str, Any]:
        total_validations = sum(len(h) for h in self._history.values())
        successful = sum(
            sum(1 for r in h if r["passed"])
            for h in self._history.values()
        )
        return {
            "total_validations": total_validations,
            "successful": successful,
            "failed": total_validations - successful,
            "algorithms_tracked": len(self._history),
            "adaptations": dict(self._adaptation_counts)
        }

    def get_algorithm_stats(self, algorithm_name: str) -> Dict[str, Any]:
        hist = self._history.get(algorithm_name, [])
        if not hist:
            return {"total": 0, "pass_rate": 0}
        passed = sum(1 for r in hist if r["passed"])
        return {
            "total": len(hist),
            "pass_rate": passed / len(hist),
            "thresholds": self.get_thresholds(algorithm_name)
        }


class ValidationResult:
    def __init__(self, passed: bool, algorithm_name: str, input_data: Any,
                 output_data: Any, error: Optional[str] = None):
        self.passed = passed
        self.algorithm_name = algorithm_name
        self.input_data = input_data
        self.output_data = output_data
        self.error = error


class PipelineValidator:
    def __init__(self):
        self.failures: List[ValidationResult] = []

    def validate_step(self, algorithm: Algorithm, input_data: Any,
                      output_data: Any) -> ValidationResult:
        try:
            valid = algorithm.validate_output(input_data, output_data)
            result = ValidationResult(
                passed=valid,
                algorithm_name=algorithm.name,
                input_data=input_data,
                output_data=output_data,
                error=None if valid else f"Validation failed for {algorithm.name}"
            )
        except Exception as e:
            result = ValidationResult(
                passed=False,
                algorithm_name=algorithm.name,
                input_data=input_data,
                output_data=output_data,
                error=str(e)
            )

        if not result.passed:
            self.failures.append(result)

        return result

    def execute_with_rollback(self, pipeline: List[Algorithm], data: Any,
                              fallback_algo: Optional[Algorithm] = None) -> Any:
        result = data
        intermediate_results = [(None, data)]

        for algo in pipeline:
            candidate = algo.process(result)
            validation = self.validate_step(algo, result, candidate)

            if validation.passed:
                intermediate_results.append((algo, candidate))
                result = candidate
            else:
                if fallback_algo:
                    result = fallback_algo.process(data)
                    intermediate_results.append((fallback_algo, result))
                return result, intermediate_results, validation

        return result, intermediate_results, None

    def get_failure_stats(self) -> Dict[str, Any]:
        if not self.failures:
            return {"total_failures": 0}

        from collections import Counter
        algo_failures = Counter(r.algorithm_name for r in self.failures)
        return {
            "total_failures": len(self.failures),
            "by_algorithm": dict(algo_failures),
            "last_failure": self.failures[-1].error if self.failures else None
        }

    def clear_failures(self):
        self.failures.clear()
