
from typing import Any, Dict, List, Optional, Callable
from algorithms.base import Algorithm


class PipelineNode:
    def __init__(self, name: str, algorithm: Algorithm, depends_on: Optional[List[str]] = None):
        self.name = name
        self.algorithm = algorithm
        self.depends_on = depends_on or []


class PipelineGraph:
    def __init__(self):
        self.nodes: Dict[str, PipelineNode] = {}
        self.node_order: List[str] = []

    def add_algorithm(self, name: str, algorithm: Algorithm,
                      depends_on: Optional[List[str]] = None):
        self.nodes[name] = PipelineNode(name, algorithm, depends_on)
        if name not in self.node_order:
            self.node_order.append(name)

    def validate(self) -> bool:
        visited: set = set()
        path: set = set()

        def dfs(name: str) -> bool:
            if name in path:
                return False
            if name in visited:
                return True
            if name not in self.nodes:
                return False
            path.add(name)
            node = self.nodes[name]
            for dep in node.depends_on:
                if not dfs(dep):
                    return False
            path.remove(name)
            visited.add(name)
            return True

        for name in self.nodes:
            if not dfs(name):
                return False
        return True

    def topological_sort(self) -> List[str]:
        visited: set = set()
        result: List[str] = []

        def dfs(name: str):
            if name in visited:
                return
            visited.add(name)
            node = self.nodes.get(name)
            if node:
                for dep in node.depends_on:
                    dfs(dep)
                result.append(name)

        for name in self.node_order:
            if name in self.nodes:
                dfs(name)

        return result

    def execute(self, data: Any) -> Any:
        if not self.validate():
            raise ValueError("Pipeline graph contains a cycle")

        order = self.topological_sort()
        intermediate: Dict[str, Any] = {}

        for name in order:
            node = self.nodes[name]
            predecessors = node.depends_on

            if not predecessors:
                input_data = data
            elif len(predecessors) == 1:
                input_data = intermediate.get(predecessors[0], data)
            else:
                input_data = self._merge_inputs(
                    [intermediate.get(p, data) for p in predecessors]
                )

            output = node.algorithm.process(input_data)
            intermediate[name] = output

        if not order:
            return data

        terminals = [n for n in order if not self._is_predecessor_of_any(n, order)]
        if len(terminals) == 1:
            return intermediate[terminals[0]]
        return {t: intermediate[t] for t in terminals}

    def _is_predecessor_of_any(self, name: str, order: List[str]) -> bool:
        for n in order:
            if n != name and name in self.nodes.get(n, PipelineNode("", None)).depends_on:
                return True
        return False

    def _merge_inputs(self, inputs: List[Any]) -> Any:
        dict_inputs = [d for d in inputs if isinstance(d, dict)]
        if dict_inputs:
            merged = {}
            for d in dict_inputs:
                merged.update(d)
            return merged
        return inputs[-1] if inputs else None

    def to_linear(self, algorithms: List[Algorithm]) -> 'PipelineGraph':
        graph = PipelineGraph()
        for i, algo in enumerate(algorithms):
            deps = [f"step_{i-1}"] if i > 0 else []
            graph.add_algorithm(f"step_{i}", algo, depends_on=deps)
        return graph
