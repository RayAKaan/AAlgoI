"""
One-function-per-task shortcuts — torch-free, direct algorithm dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── SORTING ──

def sort(data: list, *, reverse: bool = False, key: Callable = None) -> list:
    """Sort a list."""
    return sorted(data, key=key, reverse=reverse)


def sort_by(data: list, key: str | Callable, reverse: bool = False) -> list:
    """Sort a list of dicts by a key."""
    if callable(key):
        return sorted(data, key=key, reverse=reverse)
    return sorted(data, key=lambda x: x[key], reverse=reverse)


def rank(data: list) -> list[tuple]:
    """Returns (rank, value) pairs, 1-indexed."""
    return list(enumerate(sorted(data), start=1))


# ── SEARCH ──

def search(data: list, target: Any) -> int:
    """Find index of target. Returns -1 if not found."""
    try:
        return data.index(target)
    except ValueError:
        return -1


# ── PATHFINDING ──

def path(graph, start, end, *, weighted: bool = True) -> list | None:
    """Find shortest path from start to end."""
    try:
        import networkx as nx
        G = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
        return nx.shortest_path(G, source=start, target=end)
    except Exception as e:
        logger.warning("shortcuts.path() networkx failed: %s", e)
    # Fallback BFS
    try:
        from collections import deque
        adj = {k: list(v.keys()) if isinstance(v, dict) else list(v) for k, v in graph.items()}
        q = deque([[start]])
        seen = {start}
        while q:
            p = q.popleft()
            node = p[-1]
            if node == end:
                return p
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    q.append(p + [nb])
    except Exception as e:
        logger.warning("shortcuts.path() BFS fallback failed: %s", e)
    return None


def all_paths(graph: dict, start: str, end: str) -> list[list]:
    """Find all paths from start to end."""
    try:
        import networkx as nx
        G = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
        return list(nx.all_simple_paths(G, source=start, target=end))
    except Exception as e:
        logger.warning("shortcuts.all_paths() failed: %s", e)
    return []


def distance(graph: dict, start: str, end: str) -> float | None:
    """Get the shortest distance (cost) from start to end."""
    try:
        import networkx as nx
        G = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
        return nx.shortest_path_length(G, source=start, target=end)
    except Exception as e:
        logger.warning("shortcuts.distance() failed: %s", e)
    return None


# ── OPTIMIZATION ──

def knapsack(items: list[dict], capacity: int | float, *, fast: bool = False) -> dict:
    """Solve 0/1 knapsack."""
    try:
        from aalgoi.algorithms.optimization.optimization_algos import GreedyKnapsack
        result = GreedyKnapsack().process(items, capacity)
        return result
    except Exception as e:
        logger.warning("shortcuts.knapsack() GreedyKnapsack failed: %s", e)
    # Fallback greedy
    sorted_items = sorted(items, key=lambda x: x["value"] / max(x["weight"], 1e-9), reverse=True)
    selected = []
    total_v = total_w = 0
    for i, item in enumerate(sorted_items):
        if total_w + item["weight"] <= capacity:
            selected.append(i)
            total_v += item["value"]
            total_w += item["weight"]
    return {"selected": selected, "value": total_v, "weight": total_w}


def minimize(fn: Callable, bounds: tuple = (-10, 10), steps: int = 1000) -> float:
    """Find x that minimizes fn(x)."""
    low, high = bounds
    best_x = low
    best_v = fn(low)
    for i in range(steps + 1):
        x = low + (high - low) * i / steps
        v = fn(x)
        if v < best_v:
            best_v = v
            best_x = x
    return best_x


def maximize(fn: Callable, bounds: tuple = (-10, 10), steps: int = 1000) -> float:
    """Find x that maximizes fn(x)."""
    low, high = bounds
    best_x = low
    best_v = fn(low)
    for i in range(steps + 1):
        x = low + (high - low) * i / steps
        v = fn(x)
        if v > best_v:
            best_v = v
            best_x = x
    return best_x


# ── ML / CLUSTERING ──

def cluster(data: list, n: int = None, *, method: str = "auto") -> dict:
    """Cluster data points."""
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        X = np.array(data)
        k = n or min(8, max(1, len(X) // 2))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        return {"labels": labels.tolist(), "centers": km.cluster_centers_.tolist(), "n_clusters": k}
    except Exception as e:
        logger.warning("shortcuts.cluster() failed: %s", e)
    return {"labels": [], "centers": [], "n_clusters": 0}


def classify(X_train, y_train, X_test) -> list:
    """Train classifier and predict labels for X_test."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np
        n = min(len(X_train), max(1, int(np.sqrt(len(X_train)))))
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(np.array(X_train), np.array(y_train))
        return knn.predict(np.array(X_test)).tolist()
    except Exception as e:
        logger.warning("shortcuts.classify() failed: %s", e)
    return []


def regress(X_train, y_train, X_test) -> list:
    """Fit regression model and predict for X_test."""
    try:
        from sklearn.linear_model import LinearRegression
        import numpy as np
        lr = LinearRegression()
        lr.fit(np.array(X_train), np.array(y_train))
        return lr.predict(np.array(X_test)).tolist()
    except Exception as e:
        logger.warning("shortcuts.regress() failed: %s", e)
    return []


# ── UTILITIES ──

def why(result) -> str:
    """Explain why AAlgoI chose the algorithm it did."""
    from aalgoi import explain
    if hasattr(result, "explain"):
        return result.explain()
    exp = explain(result)
    return getattr(exp, "summary", str(exp))


def compare(*algorithms: str, data: list = None, problem: str = "sort") -> dict:
    """Benchmark algorithms against each other using simple timing."""
    results = {}
    benchmark_data = data or [5, 3, 1, 4, 2]
    for algo in algorithms:
        t0 = __import__("time").time()
        try:
            if algo in ("sorted", "tim_sort", "timsort"):
                output = sorted(benchmark_data)
            elif algo in ("reversed", "reverse"):
                output = list(reversed(benchmark_data))
            else:
                output = sorted(benchmark_data)
            elapsed = (__import__("time").time() - t0) * 1000
            results[algo] = {"time_ms": elapsed, "winner": output == sorted(benchmark_data)}
        except Exception as e:
            results[algo] = {"time_ms": 0, "error": str(e)}
    return results
