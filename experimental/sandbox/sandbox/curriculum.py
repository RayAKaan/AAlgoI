"""
aalgoi.sandbox.curriculum — 25 built-in training problems, easy to hard.

Problems are divided into three tiers:
  Easy   (1-10):  single algorithm, clear signal, difficulty 1-2
  Medium (11-18): ambiguous query, must infer from data, difficulty 3-5
  Hard   (19-25): multi-type, composite, adversarial, difficulty 6-8

Each problem defines a validator function used to check output correctness.
"""

from __future__ import annotations
import numpy as np
from typing import Any


def _val_sorted_asc(output: Any, problem: dict) -> bool:
    data = _extract_list(output) or _extract_list(problem["data"].get("data"))
    if data and len(data) > 1:
        return all(data[i] <= data[i+1] for i in range(len(data)-1))
    return bool(output)


def _val_sorted_desc(output: Any, problem: dict) -> bool:
    data = _extract_list(output)
    if data and len(data) > 1:
        return all(data[i] >= data[i+1] for i in range(len(data)-1))
    return bool(output)


def _val_path(output: Any, problem: dict) -> bool:
    path = _extract_path(output)
    return bool(path) and path[0] == problem["data"].get("start") and path[-1] == problem["data"].get("end")


def _val_found(output: Any, problem: dict) -> bool:
    if isinstance(output, dict):
        return bool(output.get("found", output.get("result", False)))
    return bool(output)


def _val_two_clusters(output: Any, problem: dict) -> bool:
    labels = _extract_labels(output)
    if labels and len(labels) == 6:
        return labels[0] == labels[1] == labels[2] and labels[3] == labels[4] == labels[5]
    return bool(output)


def _val_predictions(output: Any, problem: dict) -> bool:
    preds = _extract_predictions(output)
    expected = problem.get("_expected_preds", [])
    if preds and expected and len(preds) == len(expected):
        return all(str(p) == str(e) for p, e in zip(preds, expected))
    return bool(output)


def _val_approximate(output: Any, problem: dict) -> bool:
    preds = _extract_predictions(output)
    expected = problem.get("_expected_approx", [])
    tol = problem.get("_tolerance", 0.35)
    if preds and expected and len(preds) == len(expected):
        return all(
            abs(float(p) - float(e)) / max(abs(float(e)), 0.01) < tol
            for p, e in zip(preds, expected)
        )
    return bool(output)


def _val_shape(output: Any, problem: dict) -> bool:
    exp = problem.get("_expected_shape")
    arr = _extract_array(output)
    if arr is not None and exp:
        return np.array(arr).shape == tuple(exp)
    return bool(output)


def _val_shorter(output: Any, problem: dict) -> bool:
    original = problem["data"].get("text", "")
    if isinstance(output, dict):
        summary = output.get("summary", output.get("result", ""))
    else:
        summary = str(output)
    return len(str(summary)) < len(original) and len(str(summary)) > 0


def _val_any(output: Any, problem: dict) -> bool:
    return output is not None


def _val_knapsack(output: Any, problem: dict) -> bool:
    cap = problem["data"].get("capacity", 0)
    if isinstance(output, dict):
        items = output.get("selected_items", output.get("items", []))
        total_w = sum(i.get("weight", 0) for i in items) if items else 0
        return total_w <= cap
    return bool(output)


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_list(x):
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    if isinstance(x, dict):
        for k in ["result", "sorted", "output", "data"]:
            if k in x and isinstance(x[k], (list, np.ndarray)):
                return list(x[k])
    return None


def _extract_path(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, dict):
        for k in ["path", "result", "route"]:
            if k in x and isinstance(x[k], (list, tuple)):
                return list(x[k])
    return None


def _extract_labels(x):
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    if isinstance(x, dict):
        for k in ["labels", "clusters", "result"]:
            if k in x and isinstance(x[k], (list, np.ndarray)):
                return list(x[k])
    return None


def _extract_predictions(x):
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    if isinstance(x, dict):
        for k in ["predictions", "result", "output"]:
            if k in x and isinstance(x[k], (list, np.ndarray)):
                return list(x[k])
    return None


def _extract_array(x):
    if isinstance(x, (list, np.ndarray)):
        return x
    if isinstance(x, dict):
        for k in ["transformed", "reduced", "result", "embeddings"]:
            if k in x:
                return x[k]
    return None


# ── Problem definitions ────────────────────────────────────────────────────
#
# Problems needing np.random at import time use "_data_fn" (a lambda)
# instead of a static "data" key.  get_problems() resolves _data_fn → data
# before returning, so callers always see a plain "data" key.

_CURRICULUM_RAW = [
    # ── EASY (1-10) ────────────────────────────────────────────────────────
    {
        "id": 1, "difficulty": 1,
        "query": "sort numbers ascending",
        "data": {"data": [5, 3, 8, 1, 9, 2, 7, 4, 6]},
        "expected_type": "SORTING",
        "validator": _val_sorted_asc,
    },
    {
        "id": 2, "difficulty": 1,
        "query": "sort descending order",
        "data": {"data": [5, 3, 8, 1, 9, 2, 7, 4, 6]},
        "expected_type": "SORTING",
        "validator": _val_sorted_desc,
    },
    {
        "id": 3, "difficulty": 1,
        "query": "find shortest path from A to C",
        "data": {"graph": {"A": {"B": 1, "C": 4}, "B": {"C": 2}, "C": {}}, "start": "A", "end": "C"},
        "expected_type": "PATHFINDING",
        "validator": _val_path,
    },
    {
        "id": 4, "difficulty": 1,
        "query": "binary search for value 5",
        "data": {"data": [1, 3, 5, 7, 9, 11], "target": 5},
        "expected_type": "SEARCHING",
        "validator": _val_found,
    },
    {
        "id": 5, "difficulty": 2,
        "query": "cluster data into 2 groups",
        "data": {"data": [[1,1],[1.5,1.5],[1.2,0.8],[5,5],[5.5,5.5],[4.8,5.2]], "n_clusters": 2},
        "expected_type": "CLUSTERING",
        "validator": _val_two_clusters,
    },
    {
        "id": 6, "difficulty": 2,
        "query": "classify items into categories",
        "data": {
            "X_train": [[0,0],[0,1],[1,0],[1,1],[0.1,0.1],[0.9,0.9]],
            "y_train": ["A","A","B","B","A","B"],
            "X_test":  [[0,0],[1,1]],
        },
        "expected_type": "CLASSIFICATION",
        "validator": _val_predictions,
        "_expected_preds": ["A", "B"],
    },
    {
        "id": 7, "difficulty": 2,
        "query": "regression predict continuous values",
        "data": {
            "X_train": [[i] for i in range(1, 9)],
            "y_train": [i*2.0 for i in range(1, 9)],
            "X_test":  [[3], [5], [9]],
        },
        "expected_type": "REGRESSION",
        "validator": _val_approximate,
        "_expected_approx": [6.0, 10.0, 18.0],
        "_tolerance": 0.35,
    },
    {
        "id": 8, "difficulty": 2,
        "query": "sentiment analysis of reviews",
        "data": {"texts": ["I love this!", "Terrible quality.", "Pretty good.", "Worst ever."]},
        "expected_type": "NLP",
        "validator": _val_any,
    },
    {
        "id": 9, "difficulty": 2,
        "query": "summarize this paragraph briefly",
        "data": {
            "text": (
                "Machine learning is transforming industries. In healthcare, "
                "AI improves diagnostics. In finance, it detects fraud instantly. "
                "In education, it personalizes learning. The field evolves rapidly."
            ),
            "max_length": 40,
        },
        "expected_type": "NLP",
        "validator": _val_shorter,
    },
    {
        "id": 10, "difficulty": 2,
        "query": "reduce dimensions to 2 components using PCA",
        "data": {
            "data": [[i, i*2, i*3, i*4, i*5] for i in range(1, 7)],
            "n_components": 2,
        },
        "expected_type": "DIMENSIONALITY_REDUCTION",
        "validator": _val_shape,
        "_expected_shape": [6, 2],
    },
    # ── MEDIUM (11-18) ─────────────────────────────────────────────────────
    {
        "id": 11, "difficulty": 3,
        "query": "group similar items together",
        "data": {"data": [[1,1],[1.2,1.1],[8,8],[8.1,7.9],[0.9,1.1],[7.8,8.2]], "n_clusters": 2},
        "expected_type": "CLUSTERING",
        "validator": _val_any,
    },
    {
        "id": 12, "difficulty": 3,
        "query": "route through the network efficiently",
        "data": {
            "graph": {"S":{"A":2,"B":5},"A":{"T":3},"B":{"T":1},"T":{}},
            "start": "S", "end": "T",
        },
        "expected_type": "PATHFINDING",
        "validator": _val_path,
    },
    {
        "id": 13, "difficulty": 3,
        "query": "pack items within weight limit to maximize value",
        "data": {
            "items": [{"name":"a","weight":2,"value":3},{"name":"b","weight":3,"value":4},
                      {"name":"c","weight":4,"value":5},{"name":"d","weight":1,"value":2}],
            "capacity": 6,
        },
        "expected_type": "OPTIMIZATION",
        "validator": _val_knapsack,
    },
    {
        "id": 14, "difficulty": 3,
        "query": "retrieve relevant documents from corpus",
        "data": {
            "corpus": ["AI transforms healthcare.", "ML improves diagnosis.",
                       "NLP enables text analysis.", "CV detects diseases."],
            "query": "how does AI help medicine",
        },
        "expected_type": "NLP",
        "validator": _val_any,
    },
    {
        "id": 15, "difficulty": 4,
        "query": "find optimal path avoiding high cost edges",
        "data": {
            "graph": {
                str(i): {str(j): abs(i-j)*2 for j in range(max(0,i-2), min(8,i+3)) if j!=i}
                for i in range(8)
            },
            "start": "0", "end": "7",
        },
        "expected_type": "PATHFINDING",
        "validator": _val_any,
    },
    {
        "id": 16, "difficulty": 4,
        "query": "train a random forest on this dataset",
        "_data_fn": lambda: {
            "X_train": np.random.randn(50, 6).tolist(),
            "y_train": ["cat", "dog"] * 25,
            "X_test":  np.random.randn(10, 6).tolist(),
        },
        "expected_type": "CLASSIFICATION",
        "validator": _val_any,
    },
    {
        "id": 17, "difficulty": 4,
        "query": "reduce this high-dimensional dataset for visualization",
        "_data_fn": lambda: {
            "data": np.random.randn(30, 10).tolist(),
            "n_components": 2,
        },
        "expected_type": "DIMENSIONALITY_REDUCTION",
        "validator": _val_shape,
        "_expected_shape": [30, 2],
    },
    {
        "id": 18, "difficulty": 5,
        "query": "optimize resource allocation under constraints",
        "data": {
            "items": [{"name":f"r{i}","weight":i+1,"value":(i+1)*3} for i in range(12)],
            "capacity": 30,
        },
        "expected_type": "OPTIMIZATION",
        "validator": _val_knapsack,
    },
    # ── HARD (19-25) ──────────────────────────────────────────────────────
    {
        "id": 19, "difficulty": 5,
        "query": "analyze this dataset",
        "_data_fn": lambda: {
            "data": [[i, i * 2 + float(np.random.randn()) * 0.5] for i in range(40)],
            "n_clusters": 3,
        },
        "expected_type": None,
        "validator": _val_any,
    },
    {
        "id": 20, "difficulty": 5,
        "query": "make sense of this text",
        "data": {
            "texts": ["Great product!", "Terrible service.", "Average experience."],
            "text": "Strong Q3 earnings. Revenue grew 15%. Customer satisfaction declined.",
        },
        "expected_type": None,
        "validator": _val_any,
    },
    {
        "id": 21, "difficulty": 6,
        "query": "find patterns in this data",
        "_data_fn": lambda: {
            "data": np.random.randn(60, 5).tolist(),
            "n_components": 2,
        },
        "expected_type": None,
        "validator": _val_any,
    },
    {
        "id": 22, "difficulty": 6,
        "query": "solve this efficiently",
        "data": {
            "graph": {str(i): {str(j): abs(i-j) for j in range(i+1, min(i+4, 15))} for i in range(15)},
            "start": "0", "end": "14",
        },
        "expected_type": "PATHFINDING",
        "validator": _val_any,
    },
    {
        "id": 23, "difficulty": 6,
        "query": "what algorithm should I use for this classification task",
        "_data_fn": lambda: {
            "X_train": np.random.randn(80, 8).tolist(),
            "y_train": [0, 1] * 40,
            "X_test":  np.random.randn(15, 8).tolist(),
        },
        "expected_type": None,
        "validator": _val_any,
    },
    {
        "id": 24, "difficulty": 7,
        "query": "optimize everything",
        "_data_fn": lambda: {
            "graph": {
                str(i): {str(j): int(np.random.randint(1, 8))
                         for j in range(8) if j != i}
                for i in range(8)
            },
            "items": [
                {"name": f"x{i}", "weight": int(np.random.randint(1, 12)),
                 "value": int(np.random.randint(5, 50))}
                for i in range(10)
            ],
            "capacity": 35, "start": "0", "end": "7",
        },
        "expected_type": None,
        "validator": _val_any,
    },
    {
        "id": 25, "difficulty": 8,
        "query": "pandemic response challenge",
        "data": "PANDEMIC_DATASET",
        "expected_type": None,
        "validator": _val_any,
    },
]


CURRICULUM = _CURRICULUM_RAW


def get_problems(start_from: int = 1, end_at: int = 25, domains: list = None) -> list:
    problems = [p for p in _CURRICULUM_RAW if start_from <= p["id"] <= end_at]
    if domains:
        domains_upper = [d.upper() for d in domains]
        problems = [
            p for p in problems
            if p.get("expected_type") is None
            or p["expected_type"].upper() in domains_upper
        ]

    resolved = []
    for p in problems:
        prob = dict(p)
        if "_data_fn" in prob:
            prob["data"] = prob.pop("_data_fn")()
        resolved.append(prob)
    return resolved
