from typing import Any

from aalgoi.algorithms.base import Algorithm


def _handle_dict_input(data: Any) -> tuple[Any, Any, bool, bool]:
    if not isinstance(data, dict):
        return data, None, False, False
    items = data.get("data", data.get("items", data.get("array", [])))
    key = data.get("key", data.get("sort_key", None))
    order = data.get("order", data.get("direction", "asc"))
    if key and isinstance(items, list) and items and isinstance(items[0], dict):
        return items, key, order == "desc", True
    return data, None, False, False


def _sorting_validate_header(input_data: Any) -> Any:
    """Extract items list from dict input for validation."""
    if isinstance(input_data, dict):
        items = input_data.get("data", input_data.get("items", input_data.get("array", input_data)))
        if isinstance(items, list):
            return items
    return input_data


class QuickSort(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "quicksort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(log n)"
        self.tags = ["sorting", "comparison", "divide_and_conquer"]
        self.best_for = ["random_data", "large_arrays"]
        self.patterns = ["DivideAndConquer", "ComparisonSort"]
        self.problem_types = ["SORTING"]

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        input_data = _sorting_validate_header(input_data)
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data: Any) -> Any:
        items, key, reverse, is_dict = _handle_dict_input(data)
        if is_dict:
            return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        if len(data) <= 1:
            return data
        pivot = data[len(data)//2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.process(left) + middle + self.process(right)


class TimSort(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "timsort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(n)"
        self.tags = ["sorting", "stable", "hybrid"]
        self.best_for = ["nearly_sorted", "real_world_data"]
        self.patterns = ["Hybrid", "Stable", "ComparisonSort"]
        self.problem_types = ["SORTING"]

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        input_data = _sorting_validate_header(input_data)
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data: Any) -> Any:
        items, key, reverse, is_dict = _handle_dict_input(data)
        if is_dict:
            return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        return sorted(data)


class HeapSort(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "heap_sort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(1)"
        self.tags = ["sorting", "in_place", "heap"]
        self.best_for = ["large_arrays", "memory_constrained"]
        self.patterns = ["ComparisonSort", "InPlace"]
        self.problem_types = ["SORTING"]

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        input_data = _sorting_validate_header(input_data)
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data: Any) -> Any:
        items, key, reverse, is_dict = _handle_dict_input(data)
        if is_dict:
            return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        import heapq
        heap = data[:]
        heapq.heapify(heap)
        return [heapq.heappop(heap) for _ in range(len(heap))]


class InsertionSort(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "insertion_sort"
        self.time_complexity = "O(n^2)"
        self.space_complexity = "O(1)"
        self.tags = ["sorting", "stable", "in_place"]
        self.best_for = ["small_arrays", "nearly_sorted"]
        self.patterns = ["ComparisonSort", "Incremental"]
        self.problem_types = ["SORTING"]

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        input_data = _sorting_validate_header(input_data)
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data: Any) -> Any:
        items, key, reverse, is_dict = _handle_dict_input(data)
        if is_dict:
            return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        result = data[:]
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result


class RadixSort(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "radix_sort"
        self.time_complexity = "O(nk)"
        self.space_complexity = "O(n + k)"
        self.tags = ["sorting", "non_comparison", "integer"]
        self.best_for = ["integers", "fixed_length_keys"]
        self.patterns = ["NonComparisonSort", "DistributionBased"]
        self.problem_types = ["SORTING"]

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        input_data = _sorting_validate_header(input_data)
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data: Any) -> Any:
        items, key, reverse, is_dict = _handle_dict_input(data)
        if is_dict:
            return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        if not data:
            return data

        max_val = max(data)
        exp = 1
        result = data[:]

        while max_val // exp > 0:
            counting = [0] * 10
            output = [0] * len(result)

            for num in result:
                index = (num // exp) % 10
                counting[index] += 1

            for i in range(1, 10):
                counting[i] += counting[i - 1]

            for i in range(len(result) - 1, -1, -1):
                index = (result[i] // exp) % 10
                output[counting[index] - 1] = result[i]
                counting[index] -= 1

            result = output
            exp *= 10

        return result


class MergeSort(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "merge_sort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(n)"
        self.tags = ["sorting", "stable", "divide_and_conquer"]
        self.best_for = ["large_arrays", "stable_required"]
        self.patterns = ["DivideAndConquer", "Stable", "ComparisonSort"]
        self.problem_types = ["SORTING"]

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        input_data = _sorting_validate_header(input_data)
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data: Any) -> Any:
        items, key, reverse, is_dict = _handle_dict_input(data)
        if is_dict:
            return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        if len(data) <= 1:
            return data

        mid = len(data) // 2
        left = self.process(data[:mid])
        right = self.process(data[mid:])

        return self._merge(left, right)

    def _merge(self, left: list[Any], right: list[Any]) -> list[Any]:
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
