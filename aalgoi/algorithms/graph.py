from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


def _dict_to_graph(val: Any) -> nx.Graph | None:
    if isinstance(val, nx.Graph):
        return val
    if not isinstance(val, dict):
        return None
    try:
        G = nx.Graph()
        for node, neighbors in val.items():
            G.add_node(node)
            if isinstance(neighbors, list):
                for nb in neighbors:
                    G.add_edge(node, nb)
            elif isinstance(neighbors, dict):
                for nb, w in neighbors.items():
                    if isinstance(w, dict):
                        G.add_edge(node, nb, **w)
                    else:
                        G.add_edge(node, nb, weight=w)
            elif isinstance(neighbors, set):
                for nb in neighbors:
                    G.add_edge(node, nb)
        return G
    except Exception:
        return None


def _dict_to_digraph(val: Any) -> nx.DiGraph | None:
    if isinstance(val, nx.DiGraph):
        return val
    if isinstance(val, nx.Graph):
        DG = nx.DiGraph()
        DG.add_nodes_from(val.nodes)
        DG.add_edges_from(val.edges)
        return DG
    G = _dict_to_graph(val)
    if G is None:
        return None
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes)
    DG.add_edges_from(G.edges)
    return DG


def _get_graph_and_nodes(spec: ProblemSpec) -> tuple[nx.Graph, Any, Any]:
    G = nx.Graph()
    start = end = None
    for key, val in spec.inputs.items():
        if key in ("start", "source"):
            start = val
        elif key in ("end", "target", "sink"):
            end = val
        else:
            converted = _dict_to_graph(val)
            if converted is not None:
                G = converted
    return G, start, end


def _get_graph_digraph(spec: ProblemSpec) -> nx.DiGraph:
    for val in spec.inputs.values():
        converted = _dict_to_digraph(val)
        if converted is not None:
            return converted
    return nx.DiGraph()


def _get_graph(spec: ProblemSpec) -> nx.Graph:
    for val in spec.inputs.values():
        converted = _dict_to_graph(val)
        if converted is not None:
            return converted
    return nx.Graph()


@algorithm(AlgorithmSpec(
    name="bfs",
    task=ProblemTask.BFS,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V+E)", "O(V)", "V+E", "V"),
    principles=frozenset({"graph_traversal"}),
    deterministic=True, exact=True,
))
class BFS(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G, start, _ = _get_graph_and_nodes(spec)
        if start is None:
            start = next(iter(G.nodes), None)
        if start is None:
            return []
        visited = []
        q = deque([start])
        seen = {start}
        while q:
            node = q.popleft()
            visited.append(node)
            for nb in G.neighbors(node):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return visited


@algorithm(AlgorithmSpec(
    name="dfs",
    task=ProblemTask.DFS,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V+E)", "O(V)", "V+E", "V"),
    principles=frozenset({"graph_traversal"}),
    deterministic=True, exact=True,
))
class DFS(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G, start, _ = _get_graph_and_nodes(spec)
        if start is None:
            start = next(iter(G.nodes), None)
        if start is None:
            return []
        visited = []
        seen = set()
        def _dfs(node: Any) -> None:
            seen.add(node)
            visited.append(node)
            for nb in G.neighbors(node):
                if nb not in seen:
                    _dfs(nb)
        _dfs(start)
        return visited


@algorithm(AlgorithmSpec(
    name="shortest_path_unweighted",
    task=ProblemTask.SHORTEST_PATH_UNWEIGHTED,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V+E)", "O(V)", "V+E", "V"),
    principles=frozenset({"graph_traversal"}),
    deterministic=True, exact=True,
))
class ShortestPathUnweighted(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G, start, end = _get_graph_and_nodes(spec)
        if start is None or end is None:
            return []
        if isinstance(G, nx.DiGraph):
            G2 = nx.Graph()
            G2.add_nodes_from(G.nodes)
            G2.add_edges_from(G.edges)
            G = G2
        try:
            return list(nx.shortest_path(G, source=start, target=end))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []


@algorithm(AlgorithmSpec(
    name="dijkstra",
    task=ProblemTask.SHORTEST_PATH_WEIGHTED,
    domain=Domain.GRAPH,
    complexity=Complexity("O((V+E) log V)", "O(V)", "(V+E) log V", "V"),
    principles=frozenset({"greedy_exchange", "optimal_substructure"}),
    deterministic=True, exact=True,
    preconditions=(
    ),
))
class Dijkstra(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G, start, end = _get_graph_and_nodes(spec)
        if start is None or end is None:
            return []
        if isinstance(G, nx.DiGraph):
            pass
        elif not isinstance(G, nx.DiGraph):
            DG = nx.DiGraph()
            DG.add_nodes_from(G.nodes)
            for u, v, d in G.edges(data=True):
                w = d.get("weight", 1)
                DG.add_edge(u, v, weight=w)
                DG.add_edge(v, u, weight=w)
            G = DG
        try:
            path = nx.shortest_path(G, source=start, target=end, weight="weight")
            length = nx.shortest_path_length(G, source=start, target=end, weight="weight")
            return {"path": list(path), "length": length}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"path": [], "length": float("inf")}


@algorithm(AlgorithmSpec(
    name="bellman_ford",
    task=ProblemTask.SHORTEST_PATH_NEGATIVE,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V*E)", "O(V)", "V*E", "V"),
    principles=frozenset({"optimal_substructure", "dynamic_programming"}),
    deterministic=True, exact=True,
))
class BellmanFord(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G, start, end = _get_graph_and_nodes(spec)
        if start is None or end is None:
            return []
        DG = G if isinstance(G, nx.DiGraph) else nx.DiGraph(G)
        try:
            path = nx.shortest_path(DG, source=start, target=end, weight="weight", method="bellman-ford")
            length = nx.shortest_path_length(DG, source=start, target=end, weight="weight", method="bellman-ford")
            return {"path": list(path), "length": length}
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXUnbounded):
            return {"path": [], "length": float("inf")}


@algorithm(AlgorithmSpec(
    name="topological_sort",
    task=ProblemTask.TOPOLOGICAL_SORT,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V+E)", "O(V)", "V+E", "V"),
    principles=frozenset({"graph_traversal"}),
    deterministic=True, exact=True,
))
class TopologicalSort(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        DG = _get_graph_digraph(spec)
        try:
            return list(nx.topological_sort(DG))
        except (nx.NetworkXUnfeasible, Exception):
            return []


@algorithm(AlgorithmSpec(
    name="cycle_detection",
    task=ProblemTask.CYCLE_DETECTION,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V+E)", "O(V)", "V+E", "V"),
    principles=frozenset({"graph_traversal"}),
    deterministic=True, exact=True,
))
class CycleDetection(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G = _get_graph(spec)
        try:
            nx.find_cycle(G)
            return True
        except nx.NetworkXNoCycle:
            return False


@algorithm(AlgorithmSpec(
    name="connected_components",
    task=ProblemTask.CONNECTED_COMPONENTS,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V+E)", "O(V)", "V+E", "V"),
    principles=frozenset({"graph_traversal"}),
    deterministic=True, exact=True,
))
class ConnectedComponents(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G = _get_graph(spec)
        return [list(c) for c in nx.connected_components(G)]


@algorithm(AlgorithmSpec(
    name="kruskal_mst",
    task=ProblemTask.MST,
    domain=Domain.GRAPH,
    complexity=Complexity("O(E log E)", "O(V)", "E log E", "V"),
    principles=frozenset({"greedy_exchange", "union_find"}),
    deterministic=True, exact=True,
))
class KruskalMST(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G = _get_graph(spec)
        mst = nx.minimum_spanning_tree(G, weight="weight" if any(d.get("weight") for _, _, d in G.edges(data=True)) else None)
        edges = list(mst.edges(data=True))
        total = sum(d.get("weight", 1) for _, _, d in edges)
        return {"edges": [(u, v) for u, v, _ in edges], "total_weight": total}


@algorithm(AlgorithmSpec(
    name="edmonds_karp",
    task=ProblemTask.MAX_FLOW,
    domain=Domain.GRAPH,
    complexity=Complexity("O(V*E^2)", "O(V)", "V*E^2", "V"),
    principles=frozenset({"graph_flow_cut"}),
    deterministic=True, exact=True,
))
class EdmondsKarp(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        G = nx.DiGraph()
        source = sink = None
        for key, val in spec.inputs.items():
            if isinstance(val, dict) and key == "graph":
                for u, edges in val.items():
                    for v, cap in edges.items():
                        G.add_edge(u, v, capacity=float(cap))
            elif key in ("source", "s"):
                source = val
            elif key in ("sink", "t"):
                sink = val
        if source is None or sink is None or not G.edges:
            return {"flow_value": 0, "flow_dict": {}}
        try:
            flow_value, flow_dict = nx.maximum_flow(G, source, sink, flow_func=nx.algorithms.flow.edmonds_karp)
            return {"flow_value": flow_value, "flow_dict": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in flow_dict.items()}}
        except Exception:
            return {"flow_value": 0, "flow_dict": {}}
