from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="knapsack_fractional",
    task=ProblemTask.KNAPSACK_FRACTIONAL,
    domain=Domain.OPTIMIZATION,
    complexity=Complexity("O(n log n)", "O(n)", "n log n", "n"),
    principles=frozenset({"greedy_exchange"}),
    deterministic=True, exact=False,
    tags=frozenset({"approximate"}),
))
class KnapsackFractional(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        items = []
        capacity = 0
        for key, val in spec.inputs.items():
            if isinstance(val, list):
                items = val
            elif isinstance(val, (int, float)):
                capacity = int(val)
        if not items or capacity <= 0:
            return {"total_value": 0, "selected": []}
        scored = []
        for item in items:
            w = item.get("weight", item.get("wt", 1))
            v = item.get("value", item.get("val", 0))
            ratio = v / max(w, 1e-9)
            scored.append((ratio, w, v, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        total_value = 0.0
        remaining = capacity
        selected = []
        for ratio, w, v, item in scored:
            if w <= remaining:
                selected.append({**item, "fraction": 1.0})
                total_value += v
                remaining -= w
            else:
                fraction = remaining / max(w, 1e-9)
                selected.append({**item, "fraction": fraction})
                total_value += v * fraction
                break
        return {"total_value": total_value, "selected": selected}
