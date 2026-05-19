import numpy as np
from algorithms.base import Algorithm


class QuickSort(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "quicksort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(log n)"
        self.tags = ["sorting", "comparison", "divide_and_conquer"]
        self.best_for = ["random_data", "large_arrays"]

    def validate_output(self, input_data, output_data):
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data)//2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.process(left) + middle + self.process(right)


class TimSort(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "timsort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(n)"
        self.tags = ["sorting", "stable", "hybrid"]
        self.best_for = ["nearly_sorted", "real_world_data"]

    def validate_output(self, input_data, output_data):
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data):
        return sorted(data)


class HeapSort(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "heapsort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(1)"
        self.tags = ["sorting", "in_place", "heap"]
        self.best_for = ["large_arrays", "memory_constrained"]

    def validate_output(self, input_data, output_data):
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data):
        import heapq
        heap = data[:]
        heapq.heapify(heap)
        return [heapq.heappop(heap) for _ in range(len(heap))]


class InsertionSort(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "insertion_sort"
        self.time_complexity = "O(n^2)"
        self.space_complexity = "O(1)"
        self.tags = ["sorting", "stable", "in_place"]
        self.best_for = ["small_arrays", "nearly_sorted"]

    def validate_output(self, input_data, output_data):
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data):
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
    def __init__(self):
        super().__init__()
        self.name = "radix_sort"
        self.time_complexity = "O(nk)"
        self.space_complexity = "O(n + k)"
        self.tags = ["sorting", "non_comparison", "integer"]
        self.best_for = ["integers", "fixed_length_keys"]

    def validate_output(self, input_data, output_data):
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data):
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
    def __init__(self):
        super().__init__()
        self.name = "merge_sort"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(n)"
        self.tags = ["sorting", "stable", "divide_and_conquer"]
        self.best_for = ["large_arrays", "stable_required"]

    def validate_output(self, input_data, output_data):
        if not super().validate_output(input_data, output_data):
            return False
        if len(output_data) != len(input_data):
            return False
        return all(output_data[i] <= output_data[i+1] for i in range(len(output_data)-1))

    def process(self, data):
        if len(data) <= 1:
            return data

        mid = len(data) // 2
        left = self.process(data[:mid])
        right = self.process(data[mid:])

        return self._merge(left, right)

    def _merge(self, left, right):
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
