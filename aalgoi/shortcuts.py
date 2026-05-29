"""
One-function-per-task shortcuts.
No boilerplate. No imports beyond aalgoi.
Returns raw results, not Result objects.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_solver_instance = None


def _get_solver():
    global _solver_instance
    if _solver_instance is None:
        from core.smart_solver import SmartSolver
        _solver_instance = SmartSolver()
    return _solver_instance


# ─────────────────────────────────────────────────────────
# SORTING
# ─────────────────────────────────────────────────────────


def sort(
    data: list,
    *,
    reverse: bool = False,
    fast: bool = False,
    stable: bool = False,
    key: Callable = None,
) -> list:
    """
    Sort a list. AAlgoI picks the best algorithm for your data.

    Parameters
    ----------
    data : list
        List to sort
    reverse : bool, optional
        True -> descending order
    fast : bool, optional
        True -> prioritize speed over stability
    stable : bool, optional
        True -> preserve equal element order
    key : callable, optional
        Key function (applied before sorting).

        Note: When `key` is provided, AAlgoI's algorithm selection
        is bypassed and Python's native sorted() is used directly.
        This is because AAlgoI cannot pass arbitrary callables
        through its algorithm pipeline. All other arguments
        (reverse, fast, stable) are ignored when key is provided.

    Returns
    -------
    list
        Sorted list

    Examples
    --------
    >>> sort([3, 1, 4, 1, 5])
    [1, 1, 3, 4, 5]
    >>> sort([3, 1, 4], reverse=True)
    [4, 3, 1]
    >>> sort([3, 1, 4], fast=True)
    [1, 3, 4]
    >>> sort(["banana", "apple"], key=len)
    ['apple', 'banana']
    """
    if key is not None:
        return sorted(data, key=key, reverse=reverse)

    parts = ["sort"]
    if reverse:
        parts.append("descending")
    else:
        parts.append("ascending")
    if fast:
        parts.append("quickly")
    if stable:
        parts.append("stably")

    result = _get_solver().ask(" ".join(parts), data)
    out = result["result"] if result["success"] else sorted(data)
    if reverse:
        out = list(reversed(out))
    return out


def sort_by(data: list, key: str | Callable, reverse: bool = False) -> list:
    """
    Sort a list of dicts by a key.

    Parameters
    ----------
    data : list
        List of dicts to sort
    key : str or callable
        Dict key name, or callable to extract sort key
    reverse : bool, optional
        True -> descending order

    Returns
    -------
    list
        Sorted list

    Examples
    --------
    >>> sort_by(people, "age")
    >>> sort_by(items, lambda x: x["value"] / x["weight"])
    """
    if callable(key):
        return sorted(data, key=key, reverse=reverse)
    return sorted(data, key=lambda x: x[key], reverse=reverse)


def rank(data: list) -> list[tuple]:
    """
    Returns (rank, value) pairs, 1-indexed.

    Parameters
    ----------
    data : list
        List to rank

    Returns
    -------
    list of tuple
        [(1, smallest), (2, second), ...]

    Example
    -------
    >>> rank([30, 10, 20])
    [(1, 10), (2, 20), (3, 30)]
    """
    return list(enumerate(sort(data), start=1))


# ─────────────────────────────────────────────────────────
# PATHFINDING
# ─────────────────────────────────────────────────────────


def path(
    graph: dict,
    start: str,
    end: str,
    *,
    weighted: bool = True,
    algorithm: str = None,
) -> list | None:
    """
    Find shortest path from start to end.

    Parameters
    ----------
    graph : dict
        Adjacency dict {node: {neighbor: weight}} or {node: [neighbor]}
    start : str
        Start node name
    end : str
        End node name
    weighted : bool, optional
        True -> Dijkstra/A*, False -> BFS
    algorithm : str, optional
        Force a specific algorithm ("dijkstra", "astar", "bfs")

    Returns
    -------
    list or None
        List of nodes in path order, or None if no path exists

    Examples
    --------
    >>> path(graph, "A", "D")
    ['A', 'B', 'D']
    >>> path(graph, "A", "D", weighted=False)
    ['A', 'C', 'D']
    """
    desc = "find shortest path"
    if not weighted:
        desc = "find path unweighted"
    if algorithm:
        desc = f"find path using {algorithm}"

    data = {"graph": graph, "start": start, "end": end}
    result = _get_solver().ask(f"{desc} from {start} to {end}", data)
    return result["result"] if result["success"] else None


def all_paths(graph: dict, start: str, end: str) -> list[list]:
    """
    Find all paths from start to end (not just shortest).

    Parameters
    ----------
    graph : dict
        Adjacency dict
    start : str
        Start node
    end : str
        End node

    Returns
    -------
    list of list
        All paths from start to end

    Example
    -------
    >>> all_paths(graph, "A", "D")
    [['A', 'B', 'D'], ['A', 'C', 'D']]
    """
    data = {"graph": graph, "start": start, "end": end}
    result = _get_solver().ask(f"find all paths from {start} to {end}", data)
    return result["result"] if result["success"] else []


def distance(graph: dict, start: str, end: str) -> float | None:
    """
    Get the shortest distance (cost) from start to end.

    Parameters
    ----------
    graph : dict
        Adjacency dict
    start : str
        Start node
    end : str
        End node

    Returns
    -------
    float or None
        Shortest distance cost

    Example
    -------
    >>> distance(graph, "A", "D")
    7.0
    """
    data = {"graph": graph, "start": start, "end": end}
    result = _get_solver().ask(f"shortest distance from {start} to {end}", data)
    if result["success"] and isinstance(result["result"], dict):
        return result["result"].get("cost")
    return None


# ─────────────────────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────────────────────


def knapsack(
    items: list[dict],
    capacity: int | float,
    *,
    fast: bool = False,
) -> dict:
    """
    Solve the 0/1 knapsack problem.

    Parameters
    ----------
    items : list of dict
        List of {"value": ..., "weight": ...}
    capacity : int or float
        Maximum weight capacity
    fast : bool, optional
        True -> greedy approximation (faster, less optimal)

    Returns
    -------
    dict
        {"selected": [indices], "value": total, "weight": total}

    Example
    -------
    >>> knapsack([{"value": 60, "weight": 10}], capacity=50)
    {'selected': [0], 'value': 60, 'weight': 10}
    """
    desc = "maximize value fast" if fast else "maximize value"
    result = _get_solver().ask(desc, {"items": items, "capacity": capacity})
    return result["result"] if result["success"] else {"selected": [], "value": 0, "weight": 0}


def minimize(fn: Callable, bounds: tuple = (-10, 10), steps: int = 1000) -> float:
    """
    Find x that minimizes fn(x).

    Parameters
    ----------
    fn : callable
        Function to minimize
    bounds : tuple of (low, high), optional
        Search range
    steps : int, optional
        Number of evaluation steps

    Returns
    -------
    float
        x value that minimizes fn(x)

    Example
    -------
    >>> minimize(lambda x: (x - 3)**2 + 5)
    3.0
    """
    result = _get_solver().ask(
        "minimize this function",
        {"function": fn, "bounds": bounds, "steps": steps},
    )
    return result["result"] if result["success"] else None


def maximize(fn: Callable, bounds: tuple = (-10, 10), steps: int = 1000) -> float:
    """
    Find x that maximizes fn(x).

    Parameters
    ----------
    fn : callable
        Function to maximize
    bounds : tuple of (low, high), optional
        Search range
    steps : int, optional
        Number of evaluation steps

    Returns
    -------
    float
        x value that maximizes fn(x)

    Example
    -------
    >>> maximize(lambda x: -(x - 3)**2)
    3.0
    """
    result = _get_solver().ask(
        "maximize this function",
        {"function": fn, "bounds": bounds, "steps": steps},
    )
    return result["result"] if result["success"] else None


# ─────────────────────────────────────────────────────────
# ML / CLUSTERING
# ─────────────────────────────────────────────────────────


def cluster(
    data: list,
    n: int = None,
    *,
    method: str = "auto",
) -> dict:
    """
    Cluster data points.

    Parameters
    ----------
    data : list
        List of vectors
    n : int, optional
        Number of clusters (None -> auto-detect)
    method : str, optional
        "kmeans", "dbscan", or "auto"

    Returns
    -------
    dict
        {"labels": [...], "centers": [...], "n_clusters": int}

    Examples
    --------
    >>> cluster([[1,2],[5,8],[1.5,1.8]])
    >>> cluster(data, n=3)
    >>> cluster(data, method="dbscan")
    """
    if n:
        desc = f"cluster into {n} groups using {method}"
    else:
        desc = f"cluster using {method}"
    result = _get_solver().ask(desc, data)
    return result["result"] if result["success"] else {"labels": [], "centers": [], "n_clusters": 0}


def classify(X_train, y_train, X_test) -> list:
    """
    Train a classifier and predict labels for X_test.

    Parameters
    ----------
    X_train : list or array
        Training features
    y_train : list
        Training labels
    X_test : list or array
        Test features

    Returns
    -------
    list
        Predicted labels for X_test

    Example
    -------
    >>> classify(X_train, y_train, X_test)
    [0, 1, 0, ...]
    """
    result = _get_solver().ask(
        "classify this data",
        {"X_train": X_train, "y_train": y_train, "X_test": X_test},
    )
    return result["result"] if result["success"] else []


def regress(X_train, y_train, X_test) -> list:
    """
    Fit a regression model and predict for X_test.

    Parameters
    ----------
    X_train : list or array
        Training features
    y_train : list
        Training targets
    X_test : list or array
        Test features

    Returns
    -------
    list
        Predicted values for X_test

    Example
    -------
    >>> regress(X_train, y_train, X_test)
    [1.5, 2.3, ...]
    """
    result = _get_solver().ask(
        "fit regression model",
        {"X_train": X_train, "y_train": y_train, "X_test": X_test},
    )
    return result["result"] if result["success"] else []


def embed(words: list[str], dim: int = 100) -> dict:
    """
    Train word embeddings (Word2Vec-style).

    Parameters
    ----------
    words : list of str
        Training corpus sentences
    dim : int, optional
        Embedding dimensions (default 100)

    Returns
    -------
    dict
        Word vectors or model info

    Example
    -------
    >>> embed(["hello world", "foo bar"], dim=50)
    """
    result = _get_solver().ask(
        f"train word2vec with {dim} dimensions",
        {"corpus": words},
    )
    return result["result"] if result["success"] else {}


def reduce(data, n_components: int = 2, method: str = "pca") -> list:
    """
    Dimensionality reduction.

    Parameters
    ----------
    data : list or array
        High-dimensional data
    n_components : int, optional
        Target dimensions (default 2)
    method : str, optional
        "pca" or "tsne"

    Returns
    -------
    list
        Reduced-dimension data

    Examples
    --------
    >>> reduce(embeddings, n_components=2)
    >>> reduce(embeddings, n_components=2, method="tsne")
    """
    result = _get_solver().ask(
        f"reduce dimensions to {n_components} using {method}",
        data,
    )
    return result["result"] if result["success"] else data


# ─────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────


def blur(image, sigma: float = 1.0) -> Any:
    """
    Apply Gaussian blur to image array.

    Parameters
    ----------
    image : array-like
        Image data
    sigma : float, optional
        Blur radius (default 1.0)

    Returns
    -------
    array-like
        Blurred image

    Example
    -------
    >>> blur(img, sigma=2.0)
    """
    result = _get_solver().ask(f"blur image with sigma {sigma}", image)
    return result["result"] if result["success"] else image


def denoise(image, method: str = "bilateral") -> Any:
    """
    Denoise an image.

    Parameters
    ----------
    image : array-like
        Image data
    method : str, optional
        "bilateral" or "median"

    Returns
    -------
    array-like
        Denoised image

    Examples
    --------
    >>> denoise(img)
    >>> denoise(img, method="median")
    """
    result = _get_solver().ask(f"denoise image using {method}", image)
    return result["result"] if result["success"] else image


def edges(image) -> Any:
    """
    Detect edges in an image (Sobel).

    Parameters
    ----------
    image : array-like
        Image data

    Returns
    -------
    array-like
        Edge-detected image

    Example
    -------
    >>> edges(img)
    """
    result = _get_solver().ask("detect edges in image", image)
    return result["result"] if result["success"] else image


def enhance(image) -> Any:
    """
    Enhance image contrast (CLAHE).

    Parameters
    ----------
    image : array-like
        Image data

    Returns
    -------
    array-like
        Contrast-enhanced image

    Example
    -------
    >>> enhance(img)
    """
    result = _get_solver().ask("enhance image contrast", image)
    return result["result"] if result["success"] else image


# ─────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────


def search(data: list, target: Any) -> int:
    """
    Find the index of target in data. Returns -1 if not found.

    Parameters
    ----------
    data : list
        List to search
    target : any
        Value to find

    Returns
    -------
    int
        Index of target, or -1 if not found

    Examples
    --------
    >>> search([1, 2, 3, 4, 5], 3)
    2
    >>> search(["a", "b", "c"], "b")
    1
    """
    result = _get_solver().ask("find index of target", {"data": data, "target": target})
    return result["result"] if result["success"] else -1


# ─────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────


def why(result) -> str:
    """
    Explain why AAlgoI chose the algorithm it did.

    Parameters
    ----------
    result : Result or dict
        Result from solve() or solve_spec()

    Returns
    -------
    str
        Human-readable explanation

    Example
    -------
    >>> r = solve("sort", [3, 1, 2])
    >>> why(r)
    'Chose timsort because data is nearly sorted (O(n) best case).'
    """
    from aalgoi import explain

    exp = explain(result)
    return getattr(exp, "summary", str(exp))


def compare(*algorithms: str, data: list = None, problem: str = "sort") -> dict:
    """
    Benchmark multiple algorithms against each other.

    Parameters
    ----------
    algorithms : str
        One or more algorithm names
    data : list, optional
        Data to benchmark on
    problem : str, optional
        Problem description (default "sort")

    Returns
    -------
    dict
        {algo_name: {"time_ms": ..., "winner": ...}, ...}

    Example
    -------
    >>> compare("quicksort", "timsort", data=[5, 3, 1, 4, 2])
    """
    from aalgoi import benchmark, ProblemSpec, ProblemType

    results = {}
    for algo in algorithms:
        spec = ProblemSpec(name=algo, problem_type=ProblemType.SORTING)
        bm = benchmark(spec, data or [5, 3, 1, 4, 2])
        results[algo] = {"time_ms": bm.get("aalgoi_time_ms"), "winner": bm.get("winner")}
    return results
