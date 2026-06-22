from __future__ import annotations

import heapq
from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, Example, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="tim_sort",
    task=ProblemTask.SORT,
    domain=Domain.SORTING,
    complexity=Complexity("O(n log n)", "O(n)", "n log n", "n"),
    principles=frozenset({"divide_conquer", "optimal_substructure"}),
    tags=frozenset({"general_purpose", "stable"}),
    deterministic=True, exact=True,
))
class TimSort(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data = _get_list(spec)
        return sorted(data)


@algorithm(AlgorithmSpec(
    name="merge_sort",
    task=ProblemTask.SORT,
    domain=Domain.SORTING,
    complexity=Complexity("O(n log n)", "O(n)", "n log n", "n"),
    principles=frozenset({"divide_conquer"}),
    tags=frozenset({"stable"}),
    deterministic=True, exact=True,
))
class MergeSort(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data = _get_list(spec)
        return _merge_sort(data)


@algorithm(AlgorithmSpec(
    name="quick_sort",
    task=ProblemTask.SORT,
    domain=Domain.SORTING,
    complexity=Complexity("O(n log n)", "O(log n)", "n log n", "log n"),
    principles=frozenset({"divide_conquer", "randomized"}),
    tags=frozenset({"in_place"}),
    deterministic=False, exact=True,
))
class QuickSort(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data = _get_list(spec)
        arr = list(data)
        _quick_sort(arr, 0, len(arr) - 1)
        return arr


@algorithm(AlgorithmSpec(
    name="counting_sort",
    task=ProblemTask.COUNTING_SORT,
    domain=Domain.SORTING,
    complexity=Complexity("O(n+k)", "O(k)", "n+k", "k"),
    principles=frozenset({"hash_table"}),
    tags=frozenset({"integer_only", "stable"}),
    deterministic=True, exact=True,
))
class CountingSort(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data = _get_list(spec)
        if not data:
            return []
        arr = list(data)
        if not all(isinstance(x, int) for x in arr):
            raise TypeError("counting_sort requires integer inputs")
        lo = min(arr)
        hi = max(arr)
        offset = -lo
        counts = [0] * (hi - lo + 1)
        for x in arr:
            counts[x + offset] += 1
        out = []
        for i, count in enumerate(counts):
            out.extend([i - offset] * count)
        return out


def _get_list(spec: ProblemSpec) -> list:
    for val in spec.inputs.values():
        if isinstance(val, list):
            return val
    return []


def _merge_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = _merge_sort(arr[:mid])
    right = _merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def _quick_sort(arr: list, lo: int, hi: int) -> None:
    if lo >= hi:
        return
    p = _partition(arr, lo, hi)
    _quick_sort(arr, lo, p - 1)
    _quick_sort(arr, p + 1, hi)


def _partition(arr: list, lo: int, hi: int) -> int:
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1
