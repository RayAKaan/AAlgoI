import heapq
from collections import deque
from algorithms.base import Algorithm


class Dijkstra(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "dijkstra"
        self.time_complexity = "O(V log V)"
        self.space_complexity = "O(V)"
        self.tags = ["pathfinding", "weighted", "shortest_path"]
        self.best_for = ["weighted_graphs", "non_negative_weights"]

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
