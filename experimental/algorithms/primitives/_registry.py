from aalgoi.algorithms.primitives.base import Primitive
from aalgoi.algorithms.primitives.iterative import (
    IteratePrimitive, MapPrimitive, FilterPrimitive, ReducePrimitive, ScanPrimitive,
)
from aalgoi.algorithms.primitives.sorting import (
    PartitionPrimitive, QuickSortPrimitive, MergeSortPrimitive, HeapSortPrimitive,
)
from aalgoi.algorithms.primitives.search import (
    BinarySearchPrimitive, LinearSearchPrimitive, InterpolationSearchPrimitive,
    TwoPointerPrimitive, SlidingWindowPrimitive,
)
from aalgoi.algorithms.primitives.graph import (
    BFSPrimitive, DFSPrimitive, TopologicalSortPrimitive, UnionFindPrimitive,
)
from aalgoi.algorithms.primitives.optimization import (
    GreedyPrimitive, DynamicProgrammingPrimitive, GradientDescentPrimitive,
    RandomSearchPrimitive, BacktrackingPrimitive,
)
from aalgoi.algorithms.primitives.string import (
    LongestCommonSubsequencePrimitive, RabinKarpPrimitive,
)


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
