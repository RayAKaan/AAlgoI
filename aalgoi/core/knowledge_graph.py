import networkx as nx
from typing import Dict, List, Optional
import logging
from aalgoi.core.problem_spec import ProblemType

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

    # ── Cross-domain routing ──────────────────────────────────────────

    @staticmethod
    def _compatibility_oracle(source_type: str, target_type: str) -> bool:
        compatible_pairs = {
            "SORTING":          {"SEARCH", "TRANSFORMATION"},
            "SEARCH":            {"SORTING", "TRANSFORMATION"},
            "OPTIMIZATION":     {"SCHEDULING", "ROUTING", "DECISION"},
            "SCHEDULING":       {"OPTIMIZATION", "ROUTING"},
            "ROUTING":           {"OPTIMIZATION", "PATHFINDING", "SCHEDULING"},
            "PATHFINDING":      {"ROUTING", "OPTIMIZATION"},
            "CLASSIFICATION":   {"CLUSTERING", "REGRESSION", "ML"},
            "REGRESSION":       {"CLASSIFICATION", "ML"},
            "CLUSTERING":       {"CLASSIFICATION", "ML"},
            "ML":                {"CLASSIFICATION", "REGRESSION", "CLUSTERING", "NLP"},
            "NLP":               {"ML", "GENERATION"},
            "GENERATION":        {"NLP"},
            "IMAGE_PROCESSING": {"COMPUTER_VISION"},
            "COMPUTER_VISION": {"IMAGE_PROCESSING"},
            "TRANSFORMATION":   {"SORTING", "SEARCH"},
            "DECISION":         {"OPTIMIZATION", "CLASSIFICATION"},
        }
        return target_type in compatible_pairs.get(source_type, set())

    def find_cross_domain_candidates(
        self,
        problem_type: str,
        constraints: list,
        max_hops: int = 4,
    ) -> list:
        """
        BFS from problem_type through IS_A → APPLICABLE_TO → SOLVED_BY
        to find algorithms from other domains.

        Returns algorithm names sorted by similarity score descending,
        excluding algorithms that are directly SOLVED_BY the start type.
        """
        if not self.graph.has_node(problem_type):
            return []

        direct_algos = set(
            target for _, target, data
            in self.graph.edges(problem_type, data=True)
            if data.get("relation") == "SOLVED_BY"
        )

        source_domain = problem_type.upper()

        candidates = {}
        visited = set()
        queue = [(problem_type, 0, 1.0, source_domain)]

        while queue:
            node, hops, score, domain = queue.pop(0)

            if hops > max_hops:
                continue
            if node in visited:
                continue
            visited.add(node)

            node_type = self.graph.nodes[node].get("type")

            if node_type == "Algorithm" and node not in direct_algos:
                seen = candidates.get(node, 0.0)
                candidates[node] = max(seen, score)
                continue

            # Outgoing edges  (u=node → v=neighbor)
            for _, neighbor, data in self.graph.edges(node, data=True):
                rel = data.get("relation", "")
                weight = self._cross_domain_edge_weight(rel)
                if weight > 0 and neighbor not in visited:
                    neighbor_domain = neighbor.upper() if neighbor in [t.name for t in ProblemType] else domain
                    if hops == 0 or domain == neighbor_domain or self._compatibility_oracle(domain, neighbor_domain):
                        queue.append((neighbor, hops + 1, score * weight, neighbor_domain))

            # Incoming edges (u=neighbor → v=node)
            for neighbor, _, data in self.graph.in_edges(node, data=True):
                rel = data.get("relation", "")
                weight = self._cross_domain_edge_weight(rel)
                if weight > 0 and neighbor not in visited:
                    neighbor_domain = neighbor.upper() if neighbor in [t.name for t in ProblemType] else domain
                    if hops == 0 or domain == neighbor_domain or self._compatibility_oracle(domain, neighbor_domain):
                        queue.append((neighbor, hops + 1, score * weight, neighbor_domain))

        return sorted(candidates, key=candidates.get, reverse=True)

    @staticmethod
    def _cross_domain_edge_weight(relation: str) -> float:
        return {
            "SOLVED_BY": 1.0,
            "IS_A": 0.9,
            "APPLICABLE_TO": 0.8,
            "SIMILAR_TO": 0.6,
        }.get(relation, 0.0)

    def add_cross_domain_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 0.8,
    ):
        self._safe_add_node(source, type="Pattern" if relation == "APPLICABLE_TO" else "Problem")
        self._safe_add_node(target, type="Problem" if relation == "APPLICABLE_TO" else "Pattern")
        self.graph.add_edge(source, target, relation=relation, weight=weight)

    def _safe_add_node(self, node_name: str, type: str):
        if not self.graph.has_node(node_name):
            self.graph.add_node(node_name, type=type)
