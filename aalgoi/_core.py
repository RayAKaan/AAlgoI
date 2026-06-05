"""
aalgoi v2 — Algorithmic Mind.

Public API: Mind, solve, session, AlgorithmInfo, BenchmarkReport
"""

from __future__ import annotations

import json
import os
import time
import hashlib
import importlib
import sys
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from aalgoi._result import SolveResult
from aalgoi._data import normalize, detect_type


# ── Lazy imports ────────────────────────────────────────────────

_TORCH_AVAILABLE = None

def _torch_available() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _try_import(module_path: str, attr: str = None):
    """Try to import a module/attribute, return None on failure."""
    try:
        mod = importlib.import_module(module_path)
        if attr:
            return getattr(mod, attr, None)
        return mod
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


# ── Data classes ────────────────────────────────────────────────

@dataclass
class AlgorithmInfo:
    """Information about a known algorithm."""
    name: str
    code: str
    time_complexity: str
    space_complexity: str
    principles: list[str]
    best_for: list[str]
    discovered_by: str
    times_used: int = 0
    times_succeeded: int = 0
    performance: float = 0.0

    def __repr__(self):
        return f"AlgorithmInfo(name={self.name!r}, time={self.time_complexity})"

    def display(self):
        lines = [
            f"\U0001f4cb {self.name}",
            f"   Time:  {self.time_complexity}",
            f"   Space: {self.space_complexity}",
            f"   Principles: {', '.join(self.principles) if self.principles else 'none'}",
            f"   Best for: {', '.join(self.best_for) if self.best_for else 'general'}",
            f"   Used: {self.times_used}x ({self.times_succeeded} succeeded)",
            f"   Performance: {self.performance:.2f}",
        ]
        return "\n".join(lines)


@dataclass
class BenchmarkReport:
    """Results from a benchmark run."""
    total: int = 0
    correct: int = 0
    failed: int = 0
    errors: int = 0
    accuracy: float = 0.0
    by_domain: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)

    def __init__(self, data: dict):
        self.total = data.get("total", 0)
        self.correct = data.get("correct", 0)
        self.failed = data.get("failed", 0)
        self.errors = data.get("errors", 0)
        self.accuracy = data.get("accuracy", 0.0)
        self.by_domain = data.get("by_domain", {})
        self.problems = data.get("problems", [])

    def __repr__(self):
        pct = int(self.accuracy * 100)
        return f"\U0001f4ca Benchmark\n{self.total} problems, {self.correct} correct ({pct}%)"

    def details(self):
        return self.problems


# ── Mind ────────────────────────────────────────────────────────

class Mind:
    """
    Algorithmic Mind — learns, reasons, and discovers.

    Usage:
        mind = Mind("~/my_mind")
        result = mind.solve("sort the array", [3, 1, 4, 1, 5])
        print(result)  # [1, 1, 3, 4, 5]
    """

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path.home() / ".aalgoi" / "mind"
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._loaded = False
        self._mind = None
        self._solve_count = 0
        self._success_count = 0

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._mind = _create_mind(self.path)
        self._loaded = True

    # ── Properties ──────────────────────────────────────────────

    @property
    def algorithms(self) -> dict[str, AlgorithmInfo]:
        self._ensure_loaded()
        if self._mind is None:
            return _cold_start_algorithms()
        algos_attr = getattr(self._mind, "algorithms", None)
        if algos_attr and isinstance(algos_attr, dict):
            return algos_attr
        kg = getattr(self._mind, "knowledge_graph", None)
        if kg:
            return _kg_to_algorithms(kg)
        return _cold_start_algorithms()

    @property
    def principles(self) -> list[str]:
        self._ensure_loaded()
        if self._mind:
            p = getattr(self._mind, "principles", None)
            if p:
                return sorted(p)
        return sorted(_COLD_START_PRINCIPLES)

    @property
    def problems(self) -> list[str]:
        self._ensure_loaded()
        if self._mind:
            p = getattr(self._mind, "problems", None)
            if p:
                return sorted(p)
        return sorted(_COLD_START_PROBLEMS)

    # ── Solve ───────────────────────────────────────────────────

    def solve(self, problem_text: str, data: Any = None, **kwargs) -> SolveResult:
        self._ensure_loaded()
        normalized = normalize(data) if data is not None else None
        t0 = time.time()

        try:
            if self._mind is not None:
                solution = self._mind.solve(problem_text, normalized)
            else:
                solution = _rule_based_solve(problem_text, normalized)

            elapsed = (time.time() - t0) * 1000
            self._solve_count += 1

            if solution is not None:
                self._success_count += 1
                output = solution.output if hasattr(solution, "output") else solution
                return SolveResult(
                    output=output,
                    algorithm=getattr(solution, "algorithm", None),
                    complexity=getattr(solution, "complexity", None),
                    principle=getattr(solution, "principle", None),
                    time_ms=elapsed,
                    confidence=getattr(solution, "confidence", 0.8),
                    is_novel=getattr(solution, "is_novel", False),
                    iterations=getattr(solution, "iterations", 1),
                    code=getattr(solution, "code", None),
                )
            return SolveResult(output=None, error="No solution found", time_ms=elapsed)
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return SolveResult(output=None, error=str(e), time_ms=elapsed)

    def learn(self, problem_text: str, data: Any = None, expected: Any = None) -> SolveResult:
        result = self.solve(problem_text, data)
        if expected is not None and result.output is not None:
            if result.output != expected:
                result = SolveResult(
                    output=result.output,
                    algorithm=result.algorithm,
                    complexity=result.complexity,
                    time_ms=result.time_ms,
                    error=f"Expected {expected}, got {result.output}",
                )
        return result

    def train(self, epochs: int = 10, **kwargs) -> dict:
        self._ensure_loaded()
        if self._mind and hasattr(self._mind, "train"):
            return self._mind.train(epochs=epochs, **kwargs)
        return {"status": "no_training_available", "epochs": 0}

    def benchmark(self, **kwargs) -> BenchmarkReport:
        self._ensure_loaded()
        if self._mind and hasattr(self._mind, "benchmark"):
            result = self._mind.benchmark(**kwargs)
            if isinstance(result, dict):
                return BenchmarkReport(result)
            return result
        return BenchmarkReport({"total": 0, "correct": 0, "accuracy": 0.0})

    def checkpoint(self, name: str | None = None) -> str | None:
        self._ensure_loaded()
        if self._mind and hasattr(self._mind, "checkpoint"):
            return self._mind.checkpoint(name)
        return None

    def rollback(self, target: str = "last_good") -> dict:
        self._ensure_loaded()
        if self._mind and hasattr(self._mind, "rollback"):
            try:
                return self._mind.rollback(target)
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "no_mind_loaded"}

    def share(self) -> int:
        self._ensure_loaded()
        outbox = self.path / "outbox"
        outbox.mkdir(exist_ok=True)
        return len(list(outbox.glob("*.json")))

    def receive(self) -> dict:
        self._ensure_loaded()
        inbox = self.path / "inbox"
        inbox.mkdir(exist_ok=True)
        count = len(list(inbox.glob("*.json")))
        return {"updates_processed": count, "algorithms_imported": 0}

    def status(self) -> str:
        self._ensure_loaded()
        algos = self.algorithms
        from aalgoi._status import box
        lines = [
            f"Algorithms:  {len(algos)}",
            f"Principles:  {len(self.principles)}",
            f"Problems:    {len(self.problems)}",
            f"Solved:      {self._solve_count}",
            f"Success rate: {self._success_count / max(self._solve_count, 1) * 100:.0f}%",
        ]
        return box(lines, title="\U0001f9e0 Algorithmic Mind")


# ── Module-level API ────────────────────────────────────────────

def solve(problem_text: str, data: Any = None) -> SolveResult:
    """Solve an algorithmic problem."""
    mind = Mind()
    return mind.solve(problem_text, data)


def session(path: str | Path | None = None):
    """Create a session context manager for a Mind."""
    from aalgoi._session import MindSession
    return MindSession(path)


# ── Mind creation ──────────────────────────────────────────────

def _create_mind(path: Path):
    """Create a mind — with or without torch."""
    if _torch_available():
        create_fn = _try_import("core.mind.rl_mind", "create_mind")
        if create_fn:
            try:
                return create_fn(str(path))
            except Exception:
                pass

    # No-torch fallback: try KG-only mind
    create_fn = _try_import("core.knowledge_graph", "AlgorithmKnowledgeGraph")
    if create_fn:
        try:
            kg = create_fn()
            return _KGMind(kg, path)
        except Exception:
            pass

    return None


class _KGMind:
    """Minimal mind using only the knowledge graph — no torch required."""

    def __init__(self, kg, path: Path):
        self.knowledge_graph = kg
        self.path = path
        self.algorithms = _cold_start_algorithms()
        self.principles = _COLD_START_PRINCIPLES
        self.problems = _COLD_START_PROBLEMS

    def solve(self, problem_text: str, data: Any = None):
        return _rule_based_solve(problem_text, data)


# ── Rule-based fallback (no torch) ─────────────────────────────

def _rule_based_solve(problem_text: str, data: Any) -> dict:
    """Solve using rule-based matching — no torch required, 14+ domains."""
    text = problem_text.lower()

    # ── Sort ──
    if any(w in text for w in ["sort", "order", "arrange", "organize"]):
        if isinstance(data, list):
            reverse = any(w in text for w in ["descending", "desc", "reverse"])
            return _make_solution(sorted(data, reverse=reverse), "tim_sort", "O(n log n)", "divide_conquer")

    # ── Search ──
    if any(w in text for w in ["find target", "search", "binary search", "find index", "locate"]):
        if isinstance(data, dict):
            nums = data.get("nums", data.get("arr", []))
            target = data.get("target", data.get("value", None))
            if nums and target is not None:
                try:
                    import bisect
                    if nums == sorted(nums):
                        idx = bisect.bisect_left(nums, target)
                        if idx < len(nums) and nums[idx] == target:
                            return _make_solution(idx, "binary_search", "O(log n)", "divide_conquer")
                    idx = nums.index(target)
                    return _make_solution(idx, "linear_search", "O(n)", "exhaustive")
                except ValueError:
                    pass

    # ── GCD / LCM ──
    if "gcd" in text or "greatest common" in text:
        if isinstance(data, dict):
            a, b = data.get("a", 0), data.get("b", 0)
            import math
            return _make_solution(math.gcd(a, b), "euclidean_gcd", "O(log min(a,b))", "optimal_substructure")

    if "lcm" in text or "least common" in text:
        if isinstance(data, dict):
            a, b = data.get("a", 0), data.get("b", 0)
            import math
            return _make_solution(a * b // math.gcd(a, b), "lcm_via_gcd", "O(log min(a,b))", "optimal_substructure")

    # ── Pathfinding ──
    if any(w in text for w in ["path", "route", "navigate", "shortest", "distance"]):
        if isinstance(data, dict) and "graph" in data:
            try:
                import networkx as nx
                G = data["graph"]
                start = data.get("start", data.get("source", None))
                end = data.get("end", data.get("target", None))
                if start and end:
                    if not isinstance(G, nx.Graph):
                        G = nx.Graph(G)
                    path = nx.shortest_path(G, source=start, target=end)
                    return _make_solution(path, "bfs_shortest_path", "O(V+E)", "graph_traversal")
            except Exception:
                pass

    # ── Two Sum ──
    if "two" in text and "sum" in text:
        if isinstance(data, dict):
            nums = data.get("nums", [])
            target = data.get("target", 0)
            seen = {}
            for i, n in enumerate(nums):
                if target - n in seen:
                    return _make_solution([seen[target - n], i], "hash_complement", "O(n)", "hash_table")
                seen[n] = i

    # ── Max Subarray ──
    if ("maximum" in text and "subarray" in text) or "kadane" in text:
        if isinstance(data, list):
            best = curr = data[0]
            for x in data[1:]:
                curr = max(x, curr + x)
                best = max(best, curr)
            return _make_solution(best, "kadane", "O(n)", "dynamic_programming")

    # ── Knapsack ──
    if any(w in text for w in ["knapsack", "maximize value", "capacity"]):
        if isinstance(data, dict):
            items = data.get("items", data.get("elements", []))
            capacity = data.get("capacity", data.get("cap", None))
            if items and capacity is not None:
                try:
                    from algorithms.optimization.optimization_algos import GreedyKnapsack
                    solver = GreedyKnapsack()
                    result = solver.solve({"items": items, "capacity": capacity})
                    if result.get("valid"):
                        return _make_solution(result, "greedy_knapsack", "O(n log n)", "greedy_exchange")
                except Exception:
                    pass

    # ── Scheduling ──
    if any(w in text for w in ["schedule", "timetable", "assign", "deadline"]):
        if isinstance(data, list) and len(data) > 0:
            try:
                def _deadline_key(x):
                    if isinstance(x, dict):
                        return x.get("deadline", x.get("due", x.get("time", 0)))
                    if isinstance(x, (list, tuple)) and len(x) > 1:
                        return x[1]
                    return 0
                sorted_tasks = sorted(data, key=_deadline_key)
                return _make_solution(sorted_tasks, "earliest_deadline", "O(n log n)", "greedy_exchange")
            except Exception:
                pass

    # ── Clustering ──
    if any(w in text for w in ["cluster", "group", "kmeans", "dbscan"]):
        if isinstance(data, list) and len(data) > 0:
            try:
                from sklearn.cluster import KMeans
                import numpy as np
                X = np.array(data)
                n = min(8, max(1, len(X) // 2))
                kmeans = KMeans(n_clusters=n, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(X)
                return _make_solution(
                    {"labels": labels.tolist(), "centers": kmeans.cluster_centers_.tolist(), "n_clusters": n},
                    "kmeans", "O(n*k*d)", "expectation_maximization"
                )
            except Exception:
                pass

    # ── Classification ──
    if any(w in text for w in ["classify", "knn", "svm", "predict class"]):
        if isinstance(data, dict):
            X_train = data.get("X_train", data.get("train_x", []))
            y_train = data.get("y_train", data.get("train_y", []))
            X_test = data.get("X_test", data.get("test_x", X_train))
            if len(X_train) > 0 and len(y_train) > 0:
                try:
                    from sklearn.neighbors import KNeighborsClassifier
                    import numpy as np
                    n = min(len(X_train), max(1, int(np.sqrt(len(X_train)))))
                    knn = KNeighborsClassifier(n_neighbors=n)
                    knn.fit(np.array(X_train), np.array(y_train))
                    preds = knn.predict(np.array(X_test))
                    return _make_solution(preds.tolist(), "knn_classifier", "O(n*d)", "lazy_learning")
                except Exception:
                    pass

    # ── Regression ──
    if any(w in text for w in ["predict", "regress", "forecast", "linear regression"]):
        if isinstance(data, dict):
            X_train = data.get("X_train", data.get("train_x", []))
            y_train = data.get("y_train", data.get("train_y", []))
            X_test = data.get("X_test", data.get("test_x", X_train))
            if len(X_train) > 0 and len(y_train) > 0:
                try:
                    from sklearn.linear_model import LinearRegression
                    import numpy as np
                    lr = LinearRegression()
                    lr.fit(np.array(X_train), np.array(y_train))
                    preds = lr.predict(np.array(X_test))
                    return _make_solution(preds.tolist(), "linear_regression", "O(n*d^2)", "least_squares")
                except Exception:
                    pass

    # ── String Matching ──
    if any(w in text for w in ["string match", "pattern", "substring", "find text"]):
        if isinstance(data, dict):
            text_str = data.get("text", data.get("string", data.get("str", "")))
            pattern = data.get("pattern", data.get("sub", data.get("substring", "")))
            if text_str and pattern:
                idx = text_str.find(pattern)
                return _make_solution(idx, "naive_string_match", "O(n*m)", "exhaustive")

    # ── Image Processing ──
    if any(w in text for w in ["blur", "denoise", "edge", "image"]):
        if isinstance(data, dict) or hasattr(data, 'shape'):
            try:
                from scipy.ndimage import gaussian_filter
                import numpy as np
                img = np.array(data.get("image", data) if isinstance(data, dict) else data, dtype=float)
                result = gaussian_filter(img, sigma=1.0)
                return _make_solution(result, "gaussian_blur", "O(n)", "convolution")
            except Exception:
                pass

    # ── Primes ──
    if any(w in text for w in ["prime", "is prime", "factor"]):
        if isinstance(data, dict):
            n = data.get("n", data.get("num", data.get("number", 0)))
            if isinstance(n, (int, float)) and n > 1:
                import math
                n = int(n)
                for i in range(2, int(math.isqrt(n)) + 1):
                    if n % i == 0:
                        return _make_solution(False, "trial_division", "O(sqrt(n))", "exhaustive")
                return _make_solution(True, "trial_division", "O(sqrt(n))", "exhaustive")

    return None


def _make_solution(output, algorithm, complexity, principle, code=None):
    """Create a solution dict."""
    return type("Solution", (), {
        "output": output,
        "algorithm": algorithm,
        "complexity": complexity,
        "principle": principle,
        "confidence": 0.85,
        "is_novel": False,
        "iterations": 1,
        "code": code,
    })()


# ── Cold-start data ────────────────────────────────────────────

_COLD_START_PRINCIPLES = [
    "divide_conquer",
    "dynamic_programming",
    "greedy_exchange",
    "optimal_substructure",
    "backtracking",
    "branch_and_bound",
    "hash_table",
    "sliding_window",
    "two_pointers",
    "graph_traversal",
    "topological_order",
    "union_find",
    "binary_search",
    "monotonic_stack",
    "sparse_table",
    "segment_tree",
    "exhaustive",
    "randomized",
]

_COLD_START_PROBLEMS = [
    "SORTING",
    "SEARCHING",
    "PATHFINDING",
    "DYNAMIC_PROGRAMMING",
    "GRAPH_THEORY",
    "FLOW",
    "STRING_MATCHING",
    "NUMBER_THEORY",
    "OPTIMIZATION",
    "SCHEDULING",
    "CLUSTERING",
    "CLASSIFICATION",
    "REGRESSION",
    "NLP",
    "IMAGE_PROCESSING",
    "RECOMMENDATION",
    "ANOMALY_DETECTION",
    "TIME_SERIES",
    "DIMENSIONALITY_REDUCTION",
    "ENSEMBLE",
    "GENERATION",
    "RETRIEVAL",
    "COMPRESSION",
    "ENCRYPTION",
]


def _cold_start_algorithms() -> dict[str, AlgorithmInfo]:
    """Return cold-start algorithm set — covers all 14+ domains."""
    return {
        "tim_sort": AlgorithmInfo(
            name="tim_sort", code="sorted(data)", time_complexity="O(n log n)",
            space_complexity="O(n)", principles=["divide_conquer", "optimal_substructure"],
            best_for=["sort", "general_sort"], discovered_by="bootstrap",
        ),
        "quick_sort": AlgorithmInfo(
            name="quick_sort", code="quicksort(data)", time_complexity="O(n log n)",
            space_complexity="O(log n)", principles=["divide_conquer"],
            best_for=["sort", "average_case"], discovered_by="bootstrap",
        ),
        "merge_sort": AlgorithmInfo(
            name="merge_sort", code="mergesort(data)", time_complexity="O(n log n)",
            space_complexity="O(n)", principles=["divide_conquer"],
            best_for=["sort", "stable_sort"], discovered_by="bootstrap",
        ),
        "binary_search": AlgorithmInfo(
            name="binary_search", code="bisect_left(data, target)",
            time_complexity="O(log n)", space_complexity="O(1)",
            principles=["binary_search", "divide_conquer"],
            best_for=["search", "sorted_array"], discovered_by="bootstrap",
        ),
        "linear_search": AlgorithmInfo(
            name="linear_search", code="list.index(target)",
            time_complexity="O(n)", space_complexity="O(1)",
            principles=["exhaustive"],
            best_for=["search", "unsorted_array"], discovered_by="bootstrap",
        ),
        "dijkstra": AlgorithmInfo(
            name="dijkstra", code="dijkstra(graph, source)",
            time_complexity="O((V+E) log V)", space_complexity="O(V)",
            principles=["greedy_exchange", "optimal_substructure"],
            best_for=["shortest_path", "weighted_graph"], discovered_by="bootstrap",
        ),
        "bfs": AlgorithmInfo(
            name="bfs", code="bfs(graph, source)",
            time_complexity="O(V+E)", space_complexity="O(V)",
            principles=["graph_traversal"],
            best_for=["shortest_path", "unweighted_graph"], discovered_by="bootstrap",
        ),
        "kadane": AlgorithmInfo(
            name="kadane", code="kadane(data)",
            time_complexity="O(n)", space_complexity="O(1)",
            principles=["dynamic_programming", "optimal_substructure"],
            best_for=["maximum_subarray"], discovered_by="bootstrap",
        ),
        "euclidean_gcd": AlgorithmInfo(
            name="euclidean_gcd", code="math.gcd(a, b)",
            time_complexity="O(log min(a,b))", space_complexity="O(1)",
            principles=["optimal_substructure"],
            best_for=["gcd", "number_theory"], discovered_by="bootstrap",
        ),
        "lcm_via_gcd": AlgorithmInfo(
            name="lcm_via_gcd", code="a * b // math.gcd(a, b)",
            time_complexity="O(log min(a,b))", space_complexity="O(1)",
            principles=["optimal_substructure"],
            best_for=["lcm", "number_theory"], discovered_by="bootstrap",
        ),
        "hash_complement": AlgorithmInfo(
            name="hash_complement", code="two_sum(nums, target)",
            time_complexity="O(n)", space_complexity="O(n)",
            principles=["hash_table"],
            best_for=["two_sum", "complement_search"], discovered_by="bootstrap",
        ),
        "greedy_knapsack": AlgorithmInfo(
            name="greedy_knapsack", code="GreedyKnapsack().process(items, cap)",
            time_complexity="O(n log n)", space_complexity="O(n)",
            principles=["greedy_exchange"],
            best_for=["knapsack", "optimization"], discovered_by="bootstrap",
        ),
        "earliest_deadline": AlgorithmInfo(
            name="earliest_deadline", code="sorted(tasks, key=lambda x: x.deadline)",
            time_complexity="O(n log n)", space_complexity="O(n)",
            principles=["greedy_exchange"],
            best_for=["scheduling"], discovered_by="bootstrap",
        ),
        "kmeans": AlgorithmInfo(
            name="kmeans", code="KMeans(n_clusters=k).fit_predict(X)",
            time_complexity="O(n*k*d)", space_complexity="O(n+k*d)",
            principles=["expectation_maximization"],
            best_for=["clustering"], discovered_by="bootstrap",
        ),
        "knn_classifier": AlgorithmInfo(
            name="knn_classifier", code="KNeighborsClassifier(n).fit(X, y).predict(Xt)",
            time_complexity="O(n*d)", space_complexity="O(n*d)",
            principles=["lazy_learning"],
            best_for=["classification"], discovered_by="bootstrap",
        ),
        "linear_regression": AlgorithmInfo(
            name="linear_regression", code="LinearRegression().fit(X, y).predict(Xt)",
            time_complexity="O(n*d^2)", space_complexity="O(d^2)",
            principles=["least_squares"],
            best_for=["regression"], discovered_by="bootstrap",
        ),
        "naive_string_match": AlgorithmInfo(
            name="naive_string_match", code="text.find(pattern)",
            time_complexity="O(n*m)", space_complexity="O(1)",
            principles=["exhaustive"],
            best_for=["string_matching"], discovered_by="bootstrap",
        ),
        "gaussian_blur": AlgorithmInfo(
            name="gaussian_blur", code="gaussian_filter(img, sigma)",
            time_complexity="O(n)", space_complexity="O(n)",
            principles=["convolution"],
            best_for=["image_processing"], discovered_by="bootstrap",
        ),
        "trial_division": AlgorithmInfo(
            name="trial_division", code="is_prime(n)",
            time_complexity="O(sqrt(n))", space_complexity="O(1)",
            principles=["exhaustive"],
            best_for=["primes", "number_theory"], discovered_by="bootstrap",
        ),
    }


def _kg_to_algorithms(kg) -> dict[str, AlgorithmInfo]:
    """Convert knowledge graph algorithms to AlgorithmInfo dict."""
    result = _cold_start_algorithms()
    if hasattr(kg, "algorithms"):
        for name, algo in kg.algorithms.items():
            if isinstance(algo, AlgorithmInfo):
                result[name] = algo
    return result
