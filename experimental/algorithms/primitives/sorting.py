from typing import Any

import numpy as np

from aalgoi.algorithms.primitives.base import Primitive


class PartitionPrimitive(Primitive):
    name = "partition"
    tags = ["iteration", "splitting", "divide"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "tuple"
    combines_well_with = ["quicksort_primitive", "map"]

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and len(data) > 1:
            pivot = data[len(data) // 2]
            left = [x for x in data if x < pivot]
            middle = [x for x in data if x == pivot]
            right = [x for x in data if x > pivot]
            return (left, middle, right)
        return data


class QuickSortPrimitive(Primitive):
    name = "quicksort"
    tags = ["sorting", "divide_conquer", "in_place"]
    time_complexity = "O(n log n)"
    space_complexity = "O(log n)"
    input_type = "iterable"
    output_type = "sorted_iterable"
    best_for = ["general_sorting", "in_place_sort"]
    combines_well_with = ["binary_search", "interpolation_search", "greedy"]

    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            if len(data) <= 1:
                return list(data)
            pivot = data[len(data) // 2]
            left = [x for x in data if x < pivot]
            middle = [x for x in data if x == pivot]
            right = [x for x in data if x > pivot]
            return self.process(left) + middle + self.process(right)
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, list):
            return False
        return all(output_data[i] <= output_data[i + 1] for i in range(len(output_data) - 1))


class MergeSortPrimitive(Primitive):
    name = "mergesort"
    tags = ["sorting", "divide_conquer", "stable"]
    time_complexity = "O(n log n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "sorted_iterable"
    best_for = ["stable_sort", "external_sort", "large_data"]
    combines_well_with = ["binary_search", "interpolation_search"]

    def _merge(self, left: list, right: list) -> list:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            if len(data) <= 1:
                return list(data)
            mid = len(data) // 2
            left = self.process(data[:mid])
            right = self.process(data[mid:])
            return self._merge(left, right)
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, list):
            return False
        return all(output_data[i] <= output_data[i + 1] for i in range(len(output_data) - 1))


class HeapSortPrimitive(Primitive):
    name = "heapsort"
    tags = ["sorting", "heap", "selection"]
    time_complexity = "O(n log n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "sorted_iterable"
    best_for = ["in_place_sort", "priority_based"]
    combines_well_with = ["binary_search", "priority_queue"]

    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            arr = list(data)
            n = len(arr)
            for i in range(n // 2 - 1, -1, -1):
                self._heapify(arr, n, i)
            for i in range(n - 1, 0, -1):
                arr[i], arr[0] = arr[0], arr[i]
                self._heapify(arr, i, 0)
            return arr
        return data

    def _heapify(self, arr: list, n: int, i: int) -> None:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify(arr, n, largest)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, list):
            return False
        return all(output_data[i] <= output_data[i + 1] for i in range(len(output_data) - 1))
