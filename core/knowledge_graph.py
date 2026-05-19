import networkx as nx
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AlgorithmKnowledgeGraph:
    """Semantic knowledge base using NetworkX.

    Stores relationships between Problems, Algorithms, Patterns,
    Metrics, and Constraints as a directed graph. Used as a reasoning
    layer to filter candidates, find fallback alternatives, and
    explain selection paths.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_algorithm(self, algo_name: str, metadata: Dict):
        """Add an algorithm node and its semantic edges.

        Expected metadata keys:
          - time_complexity (str) -> HAS_COMPLEXITY edge to a Metric node
          - patterns (List[str])  -> IS_A edges to Pattern nodes
          - best_for (List[str])  -> PERFORMS_BEST_WHEN edges to Constraint nodes
        """
        self._safe_add_node(algo_name, type="Algorithm")
        nx.set_node_attributes(self.graph, {algo_name: metadata})

        complexity = metadata.get("time_complexity")
        if complexity:
            self._safe_add_node(complexity, type="Metric")
            self.graph.add_edge(algo_name, complexity, relation="HAS_COMPLEXITY")

        for p in metadata.get("patterns", []):
            self._safe_add_node(p, type="Pattern")
            self.graph.add_edge(algo_name, p, relation="IS_A")

        for bf in metadata.get("best_for", []):
            self._safe_add_node(bf, type="Constraint")
            self.graph.add_edge(algo_name, bf, relation="PERFORMS_BEST_WHEN")

    def add_problem_type(self, problem_name: str, solving_algorithms: List[str]):
        """Link a problem type to algorithms that solve it."""
        self._safe_add_node(problem_name, type="Problem")
        for algo in solving_algorithms:
            if self.graph.has_node(algo):
                self.graph.add_edge(problem_name, algo, relation="SOLVED_BY")

    def find_candidates(self, problem_type: str,
                        constraints: Optional[List[str]] = None) -> List[str]:
        """Traverse from Problem -> Algorithm, optionally filtered by constraints."""
        if not self.graph.has_node(problem_type):
            return []

        candidates = [
            target for _, target, data
            in self.graph.edges(problem_type, data=True)
            if data.get("relation") == "SOLVED_BY"
        ]

        if not constraints:
            return candidates

        valid = []
        for algo in candidates:
            algo_constraints = [
                target for _, target, data
                in self.graph.edges(algo, data=True)
                if data.get("relation") == "PERFORMS_BEST_WHEN"
            ]
            if any(c in algo_constraints for c in constraints):
                valid.append(algo)

        return valid if valid else candidates

    def find_alternatives(self, failed_algo: str) -> List[str]:
        """Find siblings sharing the same IS_A pattern (e.g. DivideAndConquer)."""
        if not self.graph.has_node(failed_algo):
            return []

        patterns = [
            target for _, target, data
            in self.graph.edges(failed_algo, data=True)
            if data.get("relation") == "IS_A"
        ]

        alternatives = set()
        for pattern in patterns:
            for pred in self.graph.predecessors(pattern):
                if (pred != failed_algo
                        and self.graph.nodes[pred].get("type") == "Algorithm"):
                    alternatives.add(pred)

        return list(alternatives)

    def explain_path(self, problem: str, algo: str) -> str:
        """Shortest semantic path from problem to algorithm."""
        try:
            path = nx.shortest_path(self.graph, source=problem, target=algo)
            return f"Path: {' → '.join(path)}"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return "No semantic path found."

    def _safe_add_node(self, node_name: str, type: str):
        if not self.graph.has_node(node_name):
            self.graph.add_node(node_name, type=type)
