from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from aalgoi import solve as _solve
from aalgoi.algorithms.registry import get_registry
from aalgoi.types import ProblemTask

logger = logging.getLogger(__name__)


def sort(data: list, *, reverse: bool = False, key: Callable | None = None) -> list:
    return sorted(data, key=key, reverse=reverse)


def sort_by(data: list, key: str | Callable, reverse: bool = False) -> list:
    if callable(key):
        return sorted(data, key=key, reverse=reverse)
    return sorted(data, key=lambda x: x[key], reverse=reverse)


def rank(data: list) -> list[tuple]:
    return list(enumerate(sorted(data), start=1))


def search(data: list, target: Any) -> int:
    try:
        return data.index(target)
    except ValueError:
        return -1


def path(graph: Any, start: Any, end: Any) -> list | None:
    try:
        import networkx as nx
        G = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
        return nx.shortest_path(G, source=start, target=end)
    except Exception as e:
        logger.warning("shortcuts.path() networkx failed: %s", e)
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
    try:
        import networkx as nx
        G = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
        return list(nx.all_simple_paths(G, source=start, target=end))
    except Exception as e:
        logger.warning("shortcuts.all_paths() failed: %s", e)
    return []


def distance(graph: dict, start: str, end: str) -> float | None:
    try:
        import networkx as nx
        G = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
        return nx.shortest_path_length(G, source=start, target=end)
    except Exception as e:
        logger.warning("shortcuts.distance() failed: %s", e)
    return None


_GREEDY_KNAPSACK_CACHE: list | None = None


def _get_greedy_knapsack():
    global _GREEDY_KNAPSACK_CACHE
    if _GREEDY_KNAPSACK_CACHE is not None:
        return _GREEDY_KNAPSACK_CACHE
    try:
        reg = get_registry()
        algo = reg.create("knapsack_fractional")
        _GREEDY_KNAPSACK_CACHE = algo
        return algo
    except Exception:
        return None


def knapsack(items: list[dict], capacity: int | float, *, fast: bool = False) -> dict:
    if fast:
        greedy = _get_greedy_knapsack()
        if greedy is not None:
            from aalgoi.types import ProblemSpec
            spec = ProblemSpec(id="knapsack", task=ProblemTask.KNAPSACK_FRACTIONAL, domain="optimization", inputs={"items": items, "capacity": capacity})
            try:
                result = greedy.run(spec)
                if isinstance(result, dict) and "selected" in result:
                    return result
            except Exception:
                pass
        indexed = list(enumerate(items))
        indexed.sort(key=lambda x: x[1]["value"] / max(x[1]["weight"], 1e-9), reverse=True)
        selected = []
        total_v = total_w = 0
        for original_idx, item in indexed:
            if total_w + item["weight"] <= capacity:
                selected.append(original_idx)
                total_v += item["value"]
                total_w += item["weight"]
        return {"selected": sorted(selected), "value": total_v, "weight": total_w}

    r = _solve("knapsack 01", {"items": items, "capacity": capacity})
    if r.ok and isinstance(r.output, dict):
        out = r.output
        selected = out.get("selected", [])
        if selected and isinstance(selected[0], dict):
            idx_map = []
            total_v = 0
            total_w = 0
            for sel_item in selected:
                try:
                    i = items.index(sel_item)
                    idx_map.append(i)
                    total_v += sel_item.get("value", 0)
                    total_w += sel_item.get("weight", 0)
                except ValueError:
                    pass
            return {"selected": sorted(idx_map), "value": total_v or out.get("max_value", 0), "weight": total_w or out.get("capacity_used", 0)}
        return {"selected": selected, "value": out.get("value", out.get("max_value", 0)), "weight": out.get("weight", out.get("capacity_used", 0))}
    raise ValueError("knapsack solve failed")


def minimize(fn: Callable, bounds: tuple = (-10, 10), steps: int = 1000) -> float:
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


def cluster(data: list, n: int | None = None) -> dict:
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        X = np.array(data)
        k = n or min(8, max(1, len(X) // 2))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        return {"labels": labels.tolist(), "centers": km.cluster_centers_.tolist(), "n_clusters": k}
    except Exception as e:
        logger.warning("shortcuts.cluster() failed: %s", e)
    return {"labels": [], "centers": [], "n_clusters": 0}


def classify(X_train: Any, y_train: Any, X_test: Any) -> list:
    try:
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier
        n = min(len(X_train), max(1, int(np.sqrt(len(X_train)))))
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(np.array(X_train), np.array(y_train))
        return knn.predict(np.array(X_test)).tolist()
    except Exception as e:
        logger.warning("shortcuts.classify() failed: %s", e)
    return []


def regress(X_train: Any, y_train: Any, X_test: Any) -> list:
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(np.array(X_train), np.array(y_train))
        return lr.predict(np.array(X_test)).tolist()
    except Exception as e:
        logger.warning("shortcuts.regress() failed: %s", e)
    return []


def why(result: Any) -> str:
    if hasattr(result, "trace") and result.trace:
        steps = [f"  {t.step}: {t.detail} ({t.time_ms:.1f}ms)" for t in result.trace]
        parts = [f"Algorithm: {result.algorithm or 'N/A'}", f"Validated: {result.validated}", f"Confidence: {result.confidence:.2f}", "", "Trace:"]
        return "\n".join(parts + steps)
    if hasattr(result, "algorithm") and result.algorithm:
        return f"Algorithm: {result.algorithm} | Validated: {result.validated} | Confidence: {getattr(result, 'confidence', 'N/A')}"
    return str(result)


def compare(*algorithms: str, data: list | None = None) -> dict:
    results = {}
    benchmark_data = data or [5, 3, 1, 4, 2]
    import time
    for algo in algorithms:
        t0 = time.time()
        try:
            if algo in ("sorted", "tim_sort", "timsort"):
                output = sorted(benchmark_data)
            elif algo in ("reversed", "reverse"):
                output = list(reversed(benchmark_data))
            else:
                output = sorted(benchmark_data)
            elapsed = (time.time() - t0) * 1000
            results[algo] = {"time_ms": elapsed, "winner": output == sorted(benchmark_data)}
        except Exception as e:
            results[algo] = {"time_ms": 0, "error": str(e)}
    return results


__all__ = [
    "sort", "sort_by", "rank", "search",
    "path", "all_paths", "distance",
    "knapsack", "minimize", "maximize",
    "cluster", "classify", "regress",
    "why", "compare",
]
