from typing import Any

from aalgoi.algorithms.primitives.base import Primitive


class BinarySearchPrimitive(Primitive):
    name = "binary_search"
    tags = ["search", "sorted", "logarithmic"]
    time_complexity = "O(log n)"
    space_complexity = "O(1)"
    input_type = "sorted_iterable"
    output_type = "scalar"
    best_for = ["sorted_data", "large_input"]
    combines_well_with = ["quicksort_primitive", "mergesort_primitive"]

    def __init__(self, target: Any = None) -> None:
        super().__init__()
        self.target = target

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.target is not None:
            lo, hi = 0, len(data) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if data[mid] == self.target:
                    return mid
                elif data[mid] < self.target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, int) and self.target is not None:
            if output_data == -1:
                return self.target not in input_data
            return 0 <= output_data < len(input_data) and input_data[output_data] == self.target
        return True


class LinearSearchPrimitive(Primitive):
    name = "linear_search"
    tags = ["search", "unsorted", "linear"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["unsorted_data", "small_input"]

    def __init__(self, target: Any = None) -> None:
        super().__init__()
        self.target = target

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.target is not None:
            for i, x in enumerate(data):
                if x == self.target:
                    return i
            return -1
        return data


class InterpolationSearchPrimitive(Primitive):
    name = "interpolation_search"
    tags = ["search", "sorted", "uniform"]
    time_complexity = "O(log log n)"
    space_complexity = "O(1)"
    input_type = "sorted_iterable"
    output_type = "scalar"
    best_for = ["uniformly_distributed", "large_sorted"]
    combines_well_with = ["quicksort", "mergesort", "heapsort"]

    def __init__(self, target: Any = None) -> None:
        super().__init__()
        self.target = target

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.target is not None:
            lo, hi = 0, len(data) - 1
            while lo <= hi and data[lo] <= self.target <= data[hi]:
                if data[hi] == data[lo]:
                    if data[lo] == self.target:
                        return lo
                    break
                pos = lo + int((hi - lo) * (self.target - data[lo]) / (data[hi] - data[lo]))
                if data[pos] == self.target:
                    return pos
                if data[pos] < self.target:
                    lo = pos + 1
                else:
                    hi = pos - 1
            return -1
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, int) and self.target is not None:
            if output_data == -1:
                return True
            return 0 <= output_data < len(input_data) and input_data[output_data] == self.target
        return True


class TwoPointerPrimitive(Primitive):
    name = "two_pointer"
    tags = ["iteration", "linear", "pair_search"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "tuple"
    best_for = ["sorted_pairs", "palindrome", "window"]
    combines_well_with = ["quicksort", "mergesort", "partition"]

    def __init__(self, target_sum: Any = None) -> None:
        super().__init__()
        self.target_sum = target_sum

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.target_sum is not None:
            sorted_data = sorted(data)
            left, right = 0, len(sorted_data) - 1
            while left < right:
                s = sorted_data[left] + sorted_data[right]
                if s == self.target_sum:
                    return (left, right, sorted_data[left], sorted_data[right])
                elif s < self.target_sum:
                    left += 1
                else:
                    right -= 1
            return None
        if isinstance(data, list):
            return (0, len(data) - 1)
        return data


class SlidingWindowPrimitive(Primitive):
    name = "sliding_window"
    tags = ["iteration", "window", "subarray"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["subarray", "substring", "contiguous"]
    combines_well_with = ["map", "reduce", "two_pointer"]

    def __init__(self, window_size: int = 3) -> None:
        super().__init__()
        self.window_size = window_size

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and len(data) >= self.window_size:
            max_sum = sum(data[:self.window_size])
            current_sum = max_sum
            for i in range(self.window_size, len(data)):
                current_sum += data[i] - data[i - self.window_size]
                if current_sum > max_sum:
                    max_sum = current_sum
            return max_sum
        return data
