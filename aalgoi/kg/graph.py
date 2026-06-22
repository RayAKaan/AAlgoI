from __future__ import annotations

import networkx as nx

from aalgoi.types import AlgorithmSpec, ProblemSpec, ProblemTask


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_algorithm(self, spec: AlgorithmSpec) -> None:
        node = f"algo:{spec.name}"
        self.graph.add_node(node, type="algorithm", spec=spec)
        task_node = f"task:{spec.task.value}"
        self._ensure_task_node(spec.task)
        self.graph.add_edge(task_node, node, relation="SOLVED_BY")
        for principle in spec.principles:
            p_node = f"principle:{principle}"
            self._safe_add_node(p_node, type="principle")
            self.graph.add_edge(node, p_node, relation="USES")
        complexity_label = spec.complexity.time
        c_node = f"complexity:{complexity_label}"
        self._safe_add_node(c_node, type="complexity")
        self.graph.add_edge(node, c_node, relation="HAS_COMPLEXITY")

    def _ensure_task_node(self, task: ProblemTask) -> None:
        node = f"task:{task.value}"
        self._safe_add_node(node, type="task")

    def _safe_add_node(self, name: str, **attrs) -> None:
        if not self.graph.has_node(name):
            self.graph.add_node(name, **attrs)

    def find_candidates(self, task: ProblemTask) -> list[str]:
        task_node = f"task:{task.value}"
        if not self.graph.has_node(task_node):
            return []
        algos = []
        for _, target, data in self.graph.out_edges(task_node, data=True):
            if data.get("relation") == "SOLVED_BY" and target.startswith("algo:"):
                algos.append(target.replace("algo:", ""))
        return algos

    def explain(self, name: str) -> str | None:
        node = f"algo:{name}"
        if not self.graph.has_node(node):
            return None
        spec: AlgorithmSpec | None = self.graph.nodes[node].get("spec")
        if spec is None:
            return None
        return f"{spec.name} ({spec.task.value}) — {spec.complexity.time} / {spec.complexity.space}"

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "algorithms": sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "algorithm"),
            "tasks": sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "task"),
        }

    def find_similar_tasks(self, spec: ProblemSpec) -> list[ProblemTask]:
        return [spec.task]
