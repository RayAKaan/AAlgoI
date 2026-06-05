
from typing import Any

import numpy as np

from aalgoi.algorithms.base import Algorithm


class Primitive(Algorithm):
    """Base class for primitive operations used in algorithm synthesis."""

    time_complexity: str = "O(1)"
    space_complexity: str = "O(1)"
    best_for: list[str] = []
    combines_well_with: list[str] = []
    input_type: str = "any"
    output_type: str = "any"

    def __init__(self):
        super().__init__()
        self.transform_fn = lambda x: x.get('data') if isinstance(x, dict) else x

    @staticmethod
    def _unwrap_data(data):
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        return data

    def can_compose_with(self, other: "Primitive") -> bool:
        return self.output_type == other.input_type or other.input_type == "any" or self.output_type == "any"

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "best_for": self.best_for,
            "combines_well_with": self.combines_well_with,
            "input_type": self.input_type,
            "output_type": self.output_type
        })
        return info


class IteratePrimitive(Primitive):
    name = "iterate"
    tags = ["iteration", "loop", "basic"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "iterable"

    def process(self, data: Any) -> Any:
        if hasattr(data, '__iter__'):
            return list(data)
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        return True


class MapPrimitive(Primitive):
    name = "map"
    tags = ["iteration", "transformation", "functional"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    combines_well_with = ["filter", "reduce"]

    def __init__(self, transform_fn=None):
        super().__init__()
        self.transform_fn = transform_fn

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.transform_fn:
            return [self.transform_fn(x) for x in data]
        if isinstance(data, list):
            return data
        return data


class FilterPrimitive(Primitive):
    name = "filter"
    tags = ["iteration", "selection", "functional"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    combines_well_with = ["map", "reduce"]

    def __init__(self, predicate_fn=None):
        super().__init__()
        self.predicate_fn = predicate_fn

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.predicate_fn:
            return [x for x in data if self.predicate_fn(x)]
        if isinstance(data, list):
            return data
        return data


class ReducePrimitive(Primitive):
    name = "reduce"
    tags = ["iteration", "aggregation", "functional"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    combines_well_with = ["map", "filter"]

    def __init__(self, reduce_fn=None, initial=None):
        super().__init__()
        self.reduce_fn = reduce_fn
        self.initial = initial

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.reduce_fn:
            result = self.initial
            for x in data:
                result = self.reduce_fn(result, x) if result is not None else x
            return result
        if isinstance(data, list):
            return len(data)
        return data


class ScanPrimitive(Primitive):
    name = "scan"
    tags = ["iteration", "accumulation", "prefix"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    combines_well_with = ["map", "filter"]

    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            result = []
            acc = 0
            for x in data:
                if isinstance(x, (int, float)):
                    acc += x
                result.append(acc)
            return result
        return data


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


class BinarySearchPrimitive(Primitive):
    name = "binary_search"
    tags = ["search", "sorted", "logarithmic"]
    time_complexity = "O(log n)"
    space_complexity = "O(1)"
    input_type = "sorted_iterable"
    output_type = "scalar"
    best_for = ["sorted_data", "large_input"]
    combines_well_with = ["quicksort_primitive", "mergesort_primitive"]

    def __init__(self, target=None):
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

    def __init__(self, target=None):
        super().__init__()
        self.target = target

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.target is not None:
            for i, x in enumerate(data):
                if x == self.target:
                    return i
            return -1
        return data


class GreedyPrimitive(Primitive):
    name = "greedy"
    tags = ["optimization", "heuristic", "fast"]
    time_complexity = "O(n log n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["sorting_based", "selection"]
    combines_well_with = ["quicksort_primitive", "heapsort_primitive"]

    def __init__(self, select_fn=None):
        super().__init__()
        self.select_fn = select_fn

    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            if self.select_fn:
                items = sorted(data, key=self.select_fn)
            else:
                items = sorted(data) if data else data
            return items
        return data


class DynamicProgrammingPrimitive(Primitive):
    name = "dynamic_programming"
    tags = ["optimization", "exact", "optimal"]
    time_complexity = "O(n²)"
    space_complexity = "O(n²)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["overlapping_subproblems", "optimal_substructure"]

    def __init__(self, dp_fn=None):
        super().__init__()
        self.dp_fn = dp_fn

    def process(self, data: Any) -> Any:
        if self.dp_fn and callable(self.dp_fn):
            return self.dp_fn(data)
        if isinstance(data, list) and len(data) > 0:
            return max(data)
        return data


class GradientDescentPrimitive(Primitive):
    name = "gradient_descent"
    tags = ["optimization", "iterative", "numerical"]
    time_complexity = "O(k*n)"
    space_complexity = "O(n)"
    input_type = "numeric_array"
    output_type = "scalar"
    best_for = ["convex", "continuous"]

    def __init__(self, learning_rate=0.01, max_iter=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def process(self, data: Any) -> Any:
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            nums = [float(x) for x in data if isinstance(x, (int, float))]
            if nums:
                result = sum(nums) / len(nums)
                return result
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

    def _heapify(self, arr, n, i):
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


class BFSPrimitive(Primitive):
    name = "bfs"
    tags = ["graph", "traversal", "level_order"]
    time_complexity = "O(V + E)"
    space_complexity = "O(V)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["graph_traversal", "shortest_path_unweighted", "level_order"]
    combines_well_with = ["dfs", "topological_sort"]

    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            start = next(iter(data.keys()), None)
            if start is None:
                return []
            visited = set()
            queue = [start]
            order = []
            visited.add(start)
            while queue:
                node = queue.pop(0)
                order.append(node)
                for neighbor in data.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return order
        if isinstance(data, list):
            return data
        return data


class DFSPrimitive(Primitive):
    name = "dfs"
    tags = ["graph", "traversal", "depth_first"]
    time_complexity = "O(V + E)"
    space_complexity = "O(V)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["graph_traversal", "cycle_detection", "topological_sort"]
    combines_well_with = ["bfs", "topological_sort"]

    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            start = next(iter(data.keys()), None)
            if start is None:
                return []
            visited = set()
            order = []

            def _dfs(node):
                visited.add(node)
                order.append(node)
                for neighbor in data.get(node, []):
                    if neighbor not in visited:
                        _dfs(neighbor)

            _dfs(start)
            return order
        if isinstance(data, list):
            return data
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

    def __init__(self, target=None):
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

    def __init__(self, target_sum=None):
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

    def __init__(self, window_size=3):
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


class TopologicalSortPrimitive(Primitive):
    name = "topological_sort"
    tags = ["graph", "ordering", "dependency"]
    time_complexity = "O(V + E)"
    space_complexity = "O(V)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["dependency_resolution", "scheduling", "ordering"]
    combines_well_with = ["bfs", "dfs", "greedy"]

    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            in_degree = {node: 0 for node in data}
            for node in data:
                for neighbor in data[node]:
                    in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
            queue = [node for node, deg in in_degree.items() if deg == 0]
            result = []
            while queue:
                node = queue.pop(0)
                result.append(node)
                for neighbor in data.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            return result if len(result) == len(data) else []
        if isinstance(data, list):
            return data
        return data


class UnionFindPrimitive(Primitive):
    name = "union_find"
    tags = ["graph", "disjoint_set", "connectivity"]
    time_complexity = "O(α(n))"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["connectivity", "cycle_detection_graph", "kruskal"]
    combines_well_with = ["greedy", "kruskal"]

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and len(data) > 0:
            n = max(max(u, v) for u, v in data) if data and isinstance(data[0], (tuple, list)) else len(data)
            if isinstance(data[0], (tuple, list)):
                parent = list(range(n + 1))
                rank = [0] * (n + 1)

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        parent[rx] = ry
                    elif rank[rx] > rank[ry]:
                        parent[ry] = rx
                    else:
                        parent[ry] = rx
                        rank[rx] += 1
                    return True

                components = n
                for u, v in data:
                    if union(u, v):
                        components -= 1
                return components
            return len(data)
        return data


class BacktrackingPrimitive(Primitive):
    name = "backtracking"
    tags = ["search", "recursive", "exhaustive"]
    time_complexity = "O(2ⁿ)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["permutations", "combinations", "constraint_satisfaction"]
    combines_well_with = ["greedy", "branch_and_bound"]

    def __init__(self, backtrack_fn=None):
        super().__init__()
        self.backtrack_fn = backtrack_fn

    def process(self, data: Any) -> Any:
        if self.backtrack_fn and callable(self.backtrack_fn):
            return self.backtrack_fn(data)
        if isinstance(data, list):
            len(data)
            result = []

            def _permute(current, remaining):
                if not remaining:
                    result.append(list(current))
                for i in range(len(remaining)):
                    _permute(current + [remaining[i]], remaining[:i] + remaining[i + 1:])

            _permute([], data)
            return result
        return data


class RandomSearchPrimitive(Primitive):
    name = "random_search"
    tags = ["search", "heuristic", "stochastic"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["large_space", "no_structure", "exploration"]
    combines_well_with = ["greedy", "simulated_annealing"]

    def __init__(self, target=None, max_iterations=1000):
        super().__init__()
        self.target = target
        self.max_iterations = max_iterations

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.target is not None:
            for _ in range(min(self.max_iterations, len(data))):
                import random
                idx = random.randint(0, len(data) - 1)
                if data[idx] == self.target:
                    return idx
            return -1
        return data


class LongestCommonSubsequencePrimitive(Primitive):
    name = "lcs"
    tags = ["string", "dynamic", "subsequence"]
    time_complexity = "O(n*m)"
    space_complexity = "O(n*m)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["sequence_matching", "diff", "bioinformatics"]
    combines_well_with = ["dynamic_programming", "string_match"]

    def process(self, data: Any) -> Any:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            a, b = data[0], data[1]
            n, m = len(a), len(b)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[n][m]
        return data


class RabinKarpPrimitive(Primitive):
    name = "rabin_karp"
    tags = ["string", "pattern", "hashing"]
    time_complexity = "O(n + m)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["pattern_matching", "string_search", "plagiarism"]
    combines_well_with = ["string_match", "lcs"]

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = pattern

    def process(self, data: Any) -> Any:
        if isinstance(data, str) and self.pattern:
            n, m = len(data), len(self.pattern)
            if m > n:
                return -1
            d, q = 256, 101
            p_hash = 0
            t_hash = 0
            h = pow(d, m - 1) % q
            for i in range(m):
                p_hash = (d * p_hash + ord(self.pattern[i])) % q
                t_hash = (d * t_hash + ord(data[i])) % q
            for i in range(n - m + 1):
                if p_hash == t_hash:
                    if data[i:i + m] == self.pattern:
                        return i
                if i < n - m:
                    t_hash = (d * (t_hash - ord(data[i]) * h) + ord(data[i + m])) % q
                    if t_hash < 0:
                        t_hash += q
            return -1
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, int) and self.pattern is not None:
            if output_data == -1:
                return True
            return 0 <= output_data < len(input_data) and input_data[output_data:output_data + len(self.pattern)] == self.pattern
        return True


PRIMITIVES = {
    "iterate": IteratePrimitive(),
    "map": MapPrimitive(),
    "filter": FilterPrimitive(),
    "reduce": ReducePrimitive(),
    "scan": ScanPrimitive(),
    "partition": PartitionPrimitive(),
    "binary_search": BinarySearchPrimitive(),
    "linear_search": LinearSearchPrimitive(),
    "greedy": GreedyPrimitive(),
    "dynamic_programming": DynamicProgrammingPrimitive(),
    "gradient_descent": GradientDescentPrimitive(),
    "quicksort": QuickSortPrimitive(),
    "mergesort": MergeSortPrimitive(),
    "heapsort": HeapSortPrimitive(),
    "bfs": BFSPrimitive(),
    "dfs": DFSPrimitive(),
    "interpolation_search": InterpolationSearchPrimitive(),
    "two_pointer": TwoPointerPrimitive(),
    "sliding_window": SlidingWindowPrimitive(),
    "topological_sort": TopologicalSortPrimitive(),
    "union_find": UnionFindPrimitive(),
    "backtracking": BacktrackingPrimitive(),
    "random_search": RandomSearchPrimitive(),
    "lcs": LongestCommonSubsequencePrimitive(),
    "rabin_karp": RabinKarpPrimitive(),
}


def get_primitive_names() -> list[str]:
    return list(PRIMITIVES.keys())


def get_composable_chain(start: str, end: str) -> list[Primitive] | None:
    if start not in PRIMITIVES or end not in PRIMITIVES:
        return None

    from collections import deque
    queue = deque()
    queue.append([PRIMITIVES[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        last = path[-1]

        if last.can_compose_with(PRIMITIVES[end]):
            return path + [PRIMITIVES[end]]

        for name, prim in PRIMITIVES.items():
            if name not in visited and last.can_compose_with(prim):
                visited.add(name)
                queue.append(path + [prim])

    return None


def compose_pipeline(primitive_names: list[str]) -> list[Primitive] | None:
    pipeline = []
    for name in primitive_names:
        if name not in PRIMITIVES:
            return None
        pipeline.append(PRIMITIVES[name])

    for i in range(len(pipeline) - 1):
        if not pipeline[i].can_compose_with(pipeline[i + 1]):
            return None

    return pipeline
