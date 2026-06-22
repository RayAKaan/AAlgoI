from typing import Any

import numpy as np

from aalgoi.algorithms.primitives.base import Primitive


class GreedyPrimitive(Primitive):
    name = "greedy"
    tags = ["optimization", "heuristic", "fast"]
    time_complexity = "O(n log n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["sorting_based", "selection"]
    combines_well_with = ["quicksort_primitive", "heapsort_primitive"]

    def __init__(self, select_fn: Any = None) -> None:
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

    def __init__(self, dp_fn: Any = None) -> None:
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

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 100) -> None:
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


class RandomSearchPrimitive(Primitive):
    name = "random_search"
    tags = ["search", "heuristic", "stochastic"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["large_space", "no_structure", "exploration"]
    combines_well_with = ["greedy", "simulated_annealing"]

    def __init__(self, target: Any = None, max_iterations: int = 1000) -> None:
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


class BacktrackingPrimitive(Primitive):
    name = "backtracking"
    tags = ["search", "recursive", "exhaustive"]
    time_complexity = "O(2ⁿ)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    best_for = ["permutations", "combinations", "constraint_satisfaction"]
    combines_well_with = ["greedy", "branch_and_bound"]

    def __init__(self, backtrack_fn: Any = None) -> None:
        super().__init__()
        self.backtrack_fn = backtrack_fn

    def process(self, data: Any) -> Any:
        if self.backtrack_fn and callable(self.backtrack_fn):
            return self.backtrack_fn(data)
        if isinstance(data, list):
            len(data)
            result = []

            def _permute(current: list, remaining: list) -> None:
                if not remaining:
                    result.append(list(current))
                for i in range(len(remaining)):
                    _permute(current + [remaining[i]], remaining[:i] + remaining[i + 1:])

            _permute([], data)
            return result
        return data
