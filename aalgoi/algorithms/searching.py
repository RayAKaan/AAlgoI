from __future__ import annotations

import bisect
from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="linear_search",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.SEARCHING,
    complexity=Complexity("O(n)", "O(1)", "n", "1"),
    principles=frozenset({"exhaustive"}),
    deterministic=True, exact=True,
))
class LinearSearch(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data, target = _get_data_and_target(spec)
        for i, val in enumerate(data):
            if val == target:
                return i
        return -1


@algorithm(AlgorithmSpec(
    name="binary_search",
    task=ProblemTask.BINARY_SEARCH,
    domain=Domain.SEARCHING,
    complexity=Complexity("O(log n)", "O(1)", "log n", "1"),
    principles=frozenset({"divide_conquer", "binary_search"}),
    deterministic=True, exact=True,
))
class BinarySearch(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data, target = _get_data_and_target(spec)
        idx = bisect.bisect_left(data, target)
        if idx < len(data) and data[idx] == target:
            return idx
        return -1


@algorithm(AlgorithmSpec(
    name="two_sum",
    task=ProblemTask.TWO_SUM,
    domain=Domain.SEARCHING,
    complexity=Complexity("O(n)", "O(n)", "n", "n"),
    principles=frozenset({"hash_table"}),
    deterministic=True, exact=True,
))
class TwoSum(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data = _get_list(spec)
        target = _get_target(spec)
        seen: dict = {}
        for i, n in enumerate(data):
            complement = target - n
            if complement in seen:
                return [seen[complement], i]
            seen[n] = i
        return []


@algorithm(AlgorithmSpec(
    name="lower_bound",
    task=ProblemTask.LOWER_BOUND,
    domain=Domain.SEARCHING,
    complexity=Complexity("O(log n)", "O(1)", "log n", "1"),
    principles=frozenset({"binary_search"}),
    deterministic=True, exact=True,
))
class LowerBound(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data, target = _get_data_and_target(spec)
        return bisect.bisect_left(data, target)


def _get_list(spec: ProblemSpec) -> list:
    for val in spec.inputs.values():
        if isinstance(val, list):
            return val
    return []


def _get_data_and_target(spec: ProblemSpec) -> tuple[list, Any]:
    data = None
    target_val = None
    for key, val in spec.inputs.items():
        if isinstance(val, list):
            data = val
        else:
            target_val = val
    if data is None:
        data = []
    return data, target_val


def _get_target(spec: ProblemSpec) -> Any:
    for val in spec.inputs.values():
        if not isinstance(val, list):
            return val
    return None
