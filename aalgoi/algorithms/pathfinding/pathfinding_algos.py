import heapq
from collections import deque

from aalgoi.algorithms.base import Algorithm


class Dijkstra(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "dijkstra"
        self.time_complexity = "O(V log V)"
        self.space_complexity = "O(V)"
        self.tags = ["pathfinding", "weighted", "shortest_path"]
        self.best_for = ["weighted_graphs", "non_negative_weights"]
        self.patterns = ["Greedy", "ShortestPath"]
        self.problem_types = ["PATHFINDING"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, (list, set))

    def process(self, data):
        graph = data['graph']
        start = data['start']
        end = data.get('end')

        pq = [(0, start, [])]
        visited = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if end and node == end:
                return path

            for neighbor, weight in graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + weight, neighbor, path))

        return [] if end else visited


class AStar(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "a_star"
        self.time_complexity = "O(E)"
        self.space_complexity = "O(V)"
        self.tags = ["pathfinding", "heuristic", "grid"]
        self.best_for = ["grid_based", "geo_spatial"]
        self.patterns = ["Heuristic", "ShortestPath"]
        self.problem_types = ["PATHFINDING"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, list)

    def process(self, data):
        return Dijkstra().process(data)


class BFSPathfinder(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "bfs_path"
        self.time_complexity = "O(V+E)"
        self.space_complexity = "O(V)"
        self.tags = ["pathfinding", "unweighted", "breadth_first"]
        self.best_for = ["unweighted_graphs", "shortest_path"]
        self.patterns = ["BreadthFirst", "ShortestPath"]
        self.problem_types = ["PATHFINDING"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, list)

    def process(self, data):
        graph = data['graph']
        start = data['start']
        end = data['end']

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            node, path = queue.popleft()
            if node == end:
                return path

            for neighbor in graph.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []


class FloydWarshall(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "floyd_warshall"
        self.time_complexity = "O(V^3)"
        self.space_complexity = "O(V^2)"
        self.tags = ["pathfinding", "all_pairs", "shortest_path", "dynamic_programming"]
        self.best_for = ["all_pairs_shortest", "dense_graph", "transitive_closure"]
        self.patterns = ["GraphTraversal", "DynamicProgramming", "AllPairs"]
        self.problem_types = ["PATHFINDING"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and "distances" in output_data

    def process(self, data):
        graph = data if isinstance(data, dict) and "graph" not in data else data.get("graph", data)
        nodes = list(graph.keys())
        for v in graph.values():
            for neighbor in v:
                if neighbor not in nodes:
                    nodes.append(neighbor)
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        INF = float("inf")
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u in graph:
            for v, w in graph[u].items():
                dist[idx[u]][idx[v]] = w
        for k in range(n):
            dk = dist[k]
            for i in range(n):
                dik = dist[i][k]
                if dik == INF:
                    continue
                di = dist[i]
                for j in range(n):
                    nd = dik + dk[j]
                    if nd < di[j]:
                        di[j] = nd
        result = {}
        for i, u in enumerate(nodes):
            result[u] = {}
            for j, v in enumerate(nodes):
                result[u][v] = dist[i][j]
        return {"distances": result}
