from typing import Any

from aalgoi.algorithms.primitives.base import Primitive


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

            def _dfs(node: Any) -> None:
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

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(x: int, y: int) -> bool:
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
