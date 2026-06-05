# aalgoi/__init__.py

"""
aalgoi — Algorithmic AI
=======================

An algorithmic mind that learns, reasons, and discovers.

Quick start:
    >>> import aalgoi
    >>> result = aalgoi.solve("sort the array", [3, 1, 4, 1, 5])
    >>> print(result)
    [1, 1, 3, 4, 5]

    >>> result.algorithm
    'tim_sort'
    >>> result.complexity
    'O(n log n)'
    >>> result.explain()

Session:
    >>> with aalgoi.session() as m:
    ...     m.solve("sort", [3, 1, 2])
    ...     m.solve("find shortest path", graph_data)
    ...     m.status()

Persistent mind:
    >>> mind = aalgoi.Mind("~/my_mind")
    >>> mind.solve("sort", [3, 1, 2])
    >>> mind.train(epochs=10)
    >>> mind.benchmark()
    >>> mind.checkpoint()
    >>> mind.share()

Any data format:
    >>> import numpy as np
    >>> aalgoi.solve("find peaks", np.array([1, 3, 2, 5, 4]))

    >>> import pandas as pd
    >>> aalgoi.solve("predict revenue", pd.read_csv("sales.csv"))

    >>> aalgoi.solve("analyze", "data/metrics.json")
"""

from aalgoi._core import (
    Mind,
    solve,
    session,
    AlgorithmInfo,
    BenchmarkReport,
)
from aalgoi._result import SolveResult
from aalgoi._data import normalize, detect_type, normalize_with_metadata
from aalgoi.shortcuts import sort, search, path, knapsack, cluster, classify, regress, compare, why

__version__ = "2.1.0"

def __getattr__(name):
    if name == "explain":
        from aalgoi.core.explainer import Explainer
        _explainer = Explainer()
        def explain(problem_text: str, data=None):
            return _explainer.explain(problem_text, data)
        return explain
    raise AttributeError(f"module 'aalgoi' has no attribute {name!r}")

__all__ = [
    "solve",
    "session",
    "Mind",
    "SolveResult",
    "normalize",
    "detect_type",
    "normalize_with_metadata",
    "sort", "search", "path", "knapsack",
    "cluster", "classify", "regress",
    "compare", "why",
    "AlgorithmInfo",
    "BenchmarkReport",
    "explain",
    "__version__",
]
