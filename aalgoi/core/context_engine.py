
from typing import Any

import numpy as np
import psutil


class ContextEngine:
    """
    AI-powered context analysis engine.
    Profiles input data, environment, and predicts optimal conditions.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.history: list[dict] = []
        self._cache: dict[str, Any] = {}

    def analyze(self, problem_text: str, data=None) -> dict:
        """Analyze a problem and return context."""
        data_profile = self._analyze_data(data) if data is not None else {}
        environment = self._analyze_environment()
        constraints = self._analyze_constraints()

        context = {
            "data_profile": data_profile,
            "environment": environment,
            "constraints": constraints,
        }

        features = self._extract_features(context)
        domain = self._classify_domain(features)
        complexity = self._estimate_complexity(features)

        safe_features = {}
        for k, v in features.items():
            if isinstance(v, list):
                safe_features[k] = tuple(v)
            elif isinstance(v, dict):
                safe_features[k] = tuple(sorted(v.items()))
            elif isinstance(v, set):
                safe_features[k] = tuple(sorted(v))
            else:
                safe_features[k] = v

        try:
            cache_key = hash(frozenset(safe_features.items()))
        except TypeError:
            cache_key = hash(problem_text)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            "domain": domain,
            "complexity": complexity,
            "features": features,
            "data_profile": data_profile,
            "task_type": "unknown",
            "suggestions": self._get_suggestions(domain, complexity),
        }

        if isinstance(data, dict) and "X_train" in data:
            result["ml_profile"] = self._analyze_ml_data(data)
            if data.get("y_train") is not None:
                y_sample = data["y_train"]
                if hasattr(y_sample, 'tolist'):
                    y_sample = y_sample.tolist()
                unique = len(set(y_sample[:1000]))
                result["task_type"] = "classification" if unique < 20 else "regression"
            else:
                result["task_type"] = "clustering"
        elif hasattr(data, 'shape') and len(getattr(data, 'shape', [])) >= 2:
            result["ml_profile"] = self._analyze_ml_data(data)
            result["task_type"] = "ml"

        known_types = {"sorting", "searching", "pathfinding", "optimization", "clustering", "classification", "regression", "scheduling", "string_matching", "image_processing", "ml", "safety", "transformation", "dimensionality_reduction"}
        if result["task_type"] not in known_types:
            result["task_type"] = "safety"

        self._cache[cache_key] = result
        self.history.append(result)
        return result

    def _classify_domain(self, features: dict) -> str:
        data_size = features.get("data_size_log", 0)
        is_sorted = features.get("is_sorted", 0)
        if is_sorted:
            return "searching"
        if data_size > 3:
            return "sorting"
        return "generic"

    def _estimate_complexity(self, features: dict) -> str:
        data_size = features.get("data_size_log", 0)
        if data_size > 6:
            return "O(n log n)"
        if data_size > 3:
            return "O(n)"
        return "O(1)"

    def _get_suggestions(self, domain: str, complexity: str) -> list[str]:
        suggestions = []
        if domain == "sorting":
            suggestions.append("Use quicksort for average-case, timsort for real-world")
        elif domain == "searching":
            suggestions.append("Use binary search if sorted, linear search otherwise")
        else:
            suggestions.append(f"Default strategy for {domain} domain")
        if "n log n" in complexity:
            suggestions.append("Consider divide-and-conquer approach")
        return suggestions

    def _analyze_data(self, data: Any) -> dict[str, Any]:
        """Deep data profiling."""
        profile = {
            "type": type(data).__name__,
            "size": None,
            "shape": None,
            "memory_bytes": None,
            "statistics": {},
            "patterns": {}
        }

        if hasattr(data, "__len__"):
            profile["size"] = len(data)

        if hasattr(data, "shape"):
            profile["shape"] = data.shape

        if hasattr(data, "nbytes"):
            profile["memory_bytes"] = data.nbytes
        elif isinstance(data, (list, tuple, set)):
            profile["memory_bytes"] = self._estimate_memory(data)

        # Statistical analysis for numeric data
        profile["statistics"] = self._compute_statistics(data)

        # Pattern detection
        profile["patterns"] = self._detect_patterns(data)

        return profile

    def _compute_statistics(self, data: Any) -> dict[str, Any]:
        """Compute statistical properties of data."""
        stats = {}

        try:
            if isinstance(data, (list, tuple)) and len(data) > 0:
                if all(isinstance(x, (int, float)) for x in data[:100]):
                    arr = np.array(data, dtype=float)
                    stats["mean"] = float(np.mean(arr))
                    stats["std"] = float(np.std(arr))
                    stats["min"] = float(np.min(arr))
                    stats["max"] = float(np.max(arr))
                    stats["range"] = stats["max"] - stats["min"]
                    stats["skewness"] = float(self._compute_skewness(arr))
                    stats["is_uniform"] = stats["std"] < (stats["range"] * 0.1) if stats["range"] > 0 else False
                    stats["has_duplicates"] = len(set(data)) < len(data)
                    stats["unique_ratio"] = len(set(data)) / len(data)

        except Exception as e:
            stats["error"] = str(e)

        return stats

    def _compute_skewness(self, arr: np.ndarray) -> float:
        """Compute skewness of distribution."""
        if len(arr) < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    def _detect_patterns(self, data: Any) -> dict[str, Any]:
        """Detect structural patterns in data."""
        patterns = {}

        if isinstance(data, list) and len(data) > 1:
            try:
                patterns["is_sorted"] = all(data[i] <= data[i+1] for i in range(min(len(data)-1, 1000)))
                patterns["is_reverse_sorted"] = all(data[i] >= data[i+1] for i in range(min(len(data)-1, 1000)))
            except TypeError:
                patterns["is_sorted"] = False
                patterns["is_reverse_sorted"] = False

            # Check for nearly sorted (few inversions)
            if len(data) <= 10000:
                try:
                    inversions = sum(1 for i in range(min(len(data)-1, 5000)) for j in range(i+1, min(len(data), i+100)) if data[i] > data[j])
                except TypeError:
                    inversions = len(data)
                patterns["inversion_count"] = inversions
                patterns["is_nearly_sorted"] = inversions < len(data) * 0.1
            else:
                # Sample for large data
                sample = data[::len(data)//1000 + 1]
                inversions = sum(1 for i in range(len(sample)-1) for j in range(i+1, min(len(sample), i+50)) if sample[i] > sample[j])
                patterns["is_nearly_sorted"] = inversions < len(sample) * 0.1

            # Check for repeated patterns
            if len(data) > 100:
                patterns["has_repeated_subsequences"] = self._check_repeated_patterns(data)

        return patterns

    def _check_repeated_patterns(self, data: list) -> bool:
        """Check if data has repeated subsequences."""
        if len(data) < 20:
            return False
        window = min(10, len(data) // 10)
        seen = set()
        for i in range(0, min(len(data) - window, 1000), window):
            sub = tuple(data[i:i+window])
            if sub in seen:
                return True
            seen.add(sub)
        return False

    def _analyze_environment(self) -> dict[str, Any]:
        """Analyze system environment."""
        env = {
            "cpu": {
                "percent_used": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
                "freq_mhz": None
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "percent_used": psutil.virtual_memory().percent
            },
            "disk": {
                "free_gb": psutil.disk_usage('/').free / (1024**3)
            }
        }

        try:
            env["cpu"]["freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else None
        except Exception:
            pass

        return env

    def _analyze_constraints(self) -> dict[str, Any]:
        """Analyze current constraints."""
        return {
            "time_budget_ms": self.config.get("time_budget_ms", 500),
            "memory_budget_mb": self.config.get("memory_budget_mb", 1024),
            "accuracy_target": self.config.get("accuracy_target", 0.95),
            "priority": self.config.get("priority", "balanced")  # speed, accuracy, balanced
        }

    def _predict_optimal_conditions(self, context: dict) -> dict[str, Any]:
        """AI-powered prediction of optimal execution conditions."""
        predictions = {
            "recommended_parallelism": 1,
            "recommended_batch_size": None,
            "risk_factors": [],
            "confidence": 0.5
        }

        env = context["environment"]
        data = context["data_profile"]

        # Predict parallelism based on CPU availability
        cpu_free = 100 - env["cpu"]["percent_used"]
        cpu_count = env["cpu"]["count"]

        if cpu_free > 70 and cpu_count > 2:
            predictions["recommended_parallelism"] = min(cpu_count - 1, 8)
        elif cpu_free > 40 and cpu_count > 1:
            predictions["recommended_parallelism"] = min(cpu_count // 2, 4)

        # Predict batch size based on memory
        mem_free_gb = env["memory"]["available_gb"]
        if mem_free_gb > 8:
            predictions["recommended_batch_size"] = "large"
        elif mem_free_gb > 2:
            predictions["recommended_batch_size"] = "medium"
        else:
            predictions["recommended_batch_size"] = "small"
            predictions["risk_factors"].append("low_memory")

        # Risk assessment
        if env["memory"]["percent_used"] > 90:
            predictions["risk_factors"].append("memory_pressure")
        if cpu_free < 20:
            predictions["risk_factors"].append("cpu_contention")

        # Confidence based on data quality
        if data["size"] is not None and data["size"] > 100:
            predictions["confidence"] = 0.8
        else:
            predictions["confidence"] = 0.6

        return predictions

    def _extract_features(self, context: dict) -> dict[str, float]:
        """Extract normalized feature vector for ML models."""
        features = {}

        data = context["data_profile"]
        env = context["environment"]

        # Data features
        features["data_size_log"] = np.log10(data["size"] + 1) if data["size"] else 0
        features["is_numeric"] = 1.0 if data["type"] in ("list", "tuple", "ndarray") else 0.0
        features["is_sorted"] = 1.0 if data.get("patterns", {}).get("is_sorted", False) else 0.0
        features["is_nearly_sorted"] = 1.0 if data.get("patterns", {}).get("is_nearly_sorted", False) else 0.0

        # Environment features
        features["cpu_free"] = (100 - env["cpu"]["percent_used"]) / 100.0
        features["mem_free_ratio"] = (100 - env["memory"]["percent_used"]) / 100.0
        features["cpu_count"] = min(env["cpu"]["count"], 16) / 16.0

        # Constraint features
        constraints = context["constraints"]
        features["time_budget_norm"] = min(constraints["time_budget_ms"] / 1000.0, 1.0)
        features["priority_speed"] = 1.0 if constraints["priority"] == "speed" else 0.0
        features["priority_accuracy"] = 1.0 if constraints["priority"] == "accuracy" else 0.0

        return features

    def _estimate_memory(self, data: list) -> int:
        """Estimate memory usage of a list."""
        try:
            import sys
            return sys.getsizeof(data)
        except Exception:
            return len(data) * 8  # rough estimate

    def _analyze_ml_data(self, data) -> dict:
        if isinstance(data, dict):
            X = data.get("X_train")
            y = data.get("y_train")
        elif hasattr(data, 'shape'):
            X = data
            y = None
        else:
            return {}

        if X is None:
            return {}

        profile = {"type": "ml"}

        if hasattr(X, 'shape') and len(X.shape) >= 2:
            n, d = X.shape[0], X.shape[1]
            profile["n_samples"] = int(n)
            profile["n_features"] = int(d)
            profile["samples_per_feature"] = float(n / max(1, d))

            try:
                from scipy import sparse
                if sparse.issparse(X):
                    profile["sparsity"] = float(1.0 - X.nnz / (n * d))
                elif hasattr(X, 'size') and X.size > 0:
                    profile["sparsity"] = float(1.0 - (np.count_nonzero(X) / X.size))
                else:
                    profile["sparsity"] = 0.0
            except ImportError:
                if hasattr(X, 'size') and X.size > 0:
                    profile["sparsity"] = float(1.0 - (np.count_nonzero(X) / X.size))

            profile["memory_mb"] = float(X.nbytes / (1024 * 1024)) if hasattr(X, 'nbytes') else 0.0

        if y is not None and hasattr(y, 'shape'):
            unique = np.unique(y)
            profile["n_classes"] = int(len(unique))

            if len(unique) < 20 and y.dtype.kind in 'biuf':
                profile["is_classification"] = True
                try:
                    counts = np.bincount(y.astype(int))
                    if len(counts) > 1 and min(counts) > 0:
                        profile["imbalance_ratio"] = float(max(counts) / min(counts))
                        profile["majority_class_pct"] = float(max(counts) / sum(counts))
                except (ValueError, TypeError):
                    pass
            else:
                profile["is_classification"] = False
                profile["target_mean"] = float(np.mean(y))
                profile["target_std"] = float(np.std(y))

        return profile

    def get_feature_vector(self, context: dict) -> list[float]:
        """Get flat feature vector for ML consumption."""
        return list(context["features"].values())
