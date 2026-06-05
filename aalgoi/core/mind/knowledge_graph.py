"""
The living memory of the algorithmic mind.

Week 2: Full implementation with seeded data.
- 80 algorithms across 16 domains
- 8 mathematical principles
- 16 base problem types
- Structural similarity search
- Failure tracking
- Persistence to disk
"""

import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import networkx as nx

from aalgoi.core.mind.kg_similarity import (
    compute_problem_similarity,
    extract_signature_dict,
)

if TYPE_CHECKING:
    from aalgoi.core.mind.cognitive_actions import CognitiveAction


@dataclass
class AlgorithmNode:
    name: str
    code: str
    time_complexity: str
    space_complexity: str
    principles: list[str]
    best_for: list[str]
    discovered_by: str
    performance_history: list[float]
    correctness_verified: bool
    times_used: int
    times_succeeded: int
    created_at: str
    parent_algorithms: list[str]


@dataclass
class ProblemNode:
    signature: str
    description: str
    domain: str
    hidden_structure: str
    constraints: dict[str, Any]
    best_algorithm: str
    solved_count: int
    difficulty: float
    solution_history: list[dict]


@dataclass
class PrincipleNode:
    name: str
    mathematical_basis: str
    applicable_to: list[str]
    complexity_guarantee: str
    success_rate: float


class EdgeType:
    SOLVED_BY = "SOLVED_BY"
    USES_PRINCIPLE = "USES_PRINCIPLE"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    BETTER_THAN = "BETTER_THAN"
    FAILS_ON = "FAILS_ON"
    APPLICABLE_WHEN = "APPLICABLE_WHEN"


class AlgorithmicKnowledgeGraph:
    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.graph = nx.MultiDiGraph()
        self._algo_index: dict[str, int] = {}
        self._index_algo: dict[int, str] = {}
        self._next_index = 0

        kg_file = self.persist_path / "kg.pkl"
        if kg_file.exists():
            self._load(kg_file)
        else:
            self._seed()

    def _save(self) -> None:
        data = {
            "graph": self.graph,
            "algo_index": self._algo_index,
            "index_algo": self._index_algo,
            "next_index": self._next_index,
        }
        tmp = self.persist_path / "kg_tmp.pkl"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        target = self.persist_path / "kg.pkl"
        tmp.replace(target)

    def _load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data["graph"]
        self._algo_index = data["algo_index"]
        self._index_algo = data["index_algo"]
        self._next_index = data["next_index"]

    def _seed(self) -> None:
        from aalgoi.core.mind.kg_seed_data import (
            get_seed_algorithms,
            get_seed_principles,
            get_seed_problems,
        )

        for algo in get_seed_algorithms():
            self._add_algorithm_node(algo)

        for principle in get_seed_principles():
            node_id = f"principle:{principle.name}"
            self.graph.add_node(node_id, data=principle, node_type="principle")

        for problem in get_seed_problems():
            node_id = f"problem:{problem.signature}"
            self.graph.add_node(node_id, data=problem, node_type="problem")
            algo_id = f"algorithm:{problem.best_algorithm}"
            if algo_id in self.graph:
                self.graph.add_edge(
                    node_id, algo_id,
                    relation=EdgeType.SOLVED_BY,
                    quality=1.0,
                )

        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "algorithm":
                continue
            algo = attrs["data"]
            for princ_name in algo.principles:
                princ_id = f"principle:{princ_name}"
                if princ_id in self.graph:
                    self.graph.add_edge(
                        node_id, princ_id,
                        relation=EdgeType.USES_PRINCIPLE,
                    )

        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "algorithm":
                continue
            algo = attrs["data"]
            for parent_name in algo.parent_algorithms:
                parent_id = f"algorithm:{parent_name}"
                if parent_id in self.graph:
                    self.graph.add_edge(
                        node_id, parent_id,
                        relation=EdgeType.DERIVED_FROM,
                    )

        problem_nodes = [
            (nid, attrs) for nid, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") == "problem"
        ]
        for i, (nid_a, attr_a) in enumerate(problem_nodes):
            for nid_b, attr_b in problem_nodes[i + 1:]:
                if attr_a["data"].domain == attr_b["data"].domain:
                    self.graph.add_edge(
                        nid_a, nid_b,
                        relation=EdgeType.SIMILAR_TO,
                        similarity=0.4,
                    )
                    self.graph.add_edge(
                        nid_b, nid_a,
                        relation=EdgeType.SIMILAR_TO,
                        similarity=0.4,
                    )

        self._save()

    def _add_algorithm_node(self, algo: AlgorithmNode) -> None:
        node_id = f"algorithm:{algo.name}"
        self.graph.add_node(node_id, data=algo, node_type="algorithm")
        idx = self._next_index
        self._algo_index[algo.name] = idx
        self._index_algo[idx] = algo.name
        self._next_index += 1

    def query_similar_problems(
        self,
        signature: str,
        top_k: int = 5,
    ) -> list[tuple[ProblemNode, float]]:
        exact_id = f"problem:{signature}"
        if exact_id in self.graph:
            node = self.graph.nodes[exact_id]["data"]
            return [(node, 1.0)]

        results = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "problem":
                continue
            prob = attrs["data"]
            similarity = self._match_unknown_signature(signature, prob)
            if similarity > 0.3:
                results.append((prob, similarity))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def _match_unknown_signature(
        self,
        signature: str,
        problem: ProblemNode,
    ) -> float:
        sig_lower = signature.lower()
        score = 0.0

        domain_keywords = {
            "integers": ["int", "array", "number", "sort", "integer"],
            "graph": ["graph", "edge", "node", "vertex", "path"],
            "text": ["text", "string", "word", "char"],
            "feature_matrix": ["matrix", "feature", "ml", "classify", "regress"],
            "numbers": ["number", "optim", "dp", "count"],
            "image": ["image", "pixel", "visual"],
            "tree": ["tree", "binary", "node"],
        }
        keywords = domain_keywords.get(problem.domain, [])
        if any(kw in sig_lower for kw in keywords):
            score += 0.4

        structure_keywords = {
            "total_order": ["sort", "order", "rank"],
            "graph_connectivity": ["path", "connect", "reach", "route"],
            "optimal_substructure": ["dp", "optim", "substructure", "overlap"],
            "greedy_exchange": ["greedy", "schedule", "interval"],
            "monotonic_feasibility": ["binary search", "feasib", "monoton"],
            "statistical_separation": ["classif", "cluster", "separat"],
            "statistical_fitting": ["regress", "fit", "predict"],
        }
        struct_kws = structure_keywords.get(problem.hidden_structure, [])
        if any(kw in sig_lower for kw in struct_kws):
            score += 0.3

        return score

    def get_best_algorithms_for(
        self,
        signature: str,
        constraints: dict,
    ) -> list[AlgorithmNode]:
        similar_problems = self.query_similar_problems(signature)
        algorithm_scores: dict[str, float] = {}

        for prob, similarity in similar_problems:
            prob_node_id = f"problem:{prob.signature}"
            for _, target_id, edge_data in self.graph.out_edges(
                prob_node_id, data=True
            ):
                if edge_data.get("relation") != EdgeType.SOLVED_BY:
                    continue
                if target_id not in self.graph:
                    continue
                algo = self.graph.nodes[target_id]["data"]
                success_rate = (
                    algo.times_succeeded / max(algo.times_used, 1)
                )
                quality = edge_data.get("quality", 0.5)
                score = similarity * (
                    0.5 * success_rate + 0.3 * quality
                    + 0.2 * (algo.performance_history[-1] if algo.performance_history else 0.5)
                )
                algorithm_scores[target_id] = max(
                    algorithm_scores.get(target_id, 0), score
                )

        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "algorithm":
                continue
            algo = attrs["data"]
            sig_lower = signature.lower()
            for tag in algo.best_for:
                if tag.lower() in sig_lower:
                    base_score = 0.3
                    success_rate = algo.times_succeeded / max(algo.times_used, 1)
                    score = base_score * (0.5 + 0.5 * success_rate)
                    algorithm_scores[node_id] = max(
                        algorithm_scores.get(node_id, 0), score
                    )

        for algo_id in list(algorithm_scores.keys()):
            failure_penalty = self._count_failures(algo_id, signature)
            algorithm_scores[algo_id] *= max(0.1, 1.0 - 0.2 * failure_penalty)

        sorted_algos = sorted(
            algorithm_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for algo_id, score in sorted_algos[:10]:
            algo = self.graph.nodes[algo_id]["data"]
            results.append(algo)

        return results

    def _count_failures(self, algo_id: str, signature: str) -> int:
        count = 0
        for _, target_id, edge_data in self.graph.out_edges(algo_id, data=True):
            if edge_data.get("relation") == EdgeType.FAILS_ON:
                if signature.lower() in target_id.lower():
                    count += 1
        return count

    def get_known_failures(self, signature: str) -> list[str]:
        similar = self.query_similar_problems(signature)
        failures = []
        for prob, _ in similar:
            prob_id = f"problem:{prob.signature}"
            for source_id, _, edge_data in self.graph.in_edges(prob_id, data=True):
                if edge_data.get("relation") == EdgeType.FAILS_ON:
                    algo_name = source_id.replace("algorithm:", "")
                    if algo_name not in failures:
                        failures.append(algo_name)
        return failures

    def record_new_algorithm(
        self,
        algorithm: AlgorithmNode,
        problem_signature: str,
        cognitive_actions: list["CognitiveAction"],
    ) -> None:
        self._add_algorithm_node(algorithm)
        algo_id = f"algorithm:{algorithm.name}"

        prob_id = f"problem:{problem_signature}"
        if prob_id not in self.graph:
            self.graph.add_node(
                prob_id,
                data=ProblemNode(
                    signature=problem_signature,
                    description="",
                    domain="unknown",
                    hidden_structure="unknown",
                    constraints={},
                    best_algorithm=algorithm.name,
                    solved_count=1,
                    difficulty=0.5,
                    solution_history=[],
                ),
                node_type="problem",
            )
        self.graph.add_edge(
            prob_id, algo_id,
            relation=EdgeType.SOLVED_BY,
            quality=algorithm.performance_history[-1] if algorithm.performance_history else 0.5,
        )

        for princ_name in algorithm.principles:
            princ_id = f"principle:{princ_name}"
            if princ_id in self.graph:
                self.graph.add_edge(
                    algo_id, princ_id,
                    relation=EdgeType.USES_PRINCIPLE,
                )

        for parent_name in algorithm.parent_algorithms:
            parent_id = f"algorithm:{parent_name}"
            if parent_id in self.graph:
                self.graph.add_edge(
                    algo_id, parent_id,
                    relation=EdgeType.DERIVED_FROM,
                )

        self.graph.nodes[algo_id]["discovery_path"] = [
            int(a) for a in cognitive_actions
        ]
        self._save()

    def record_failure(
        self,
        algorithm_name: str,
        problem_signature: str,
        failure_reason: str,
    ) -> None:
        algo_id = f"algorithm:{algorithm_name}"
        prob_id = f"problem:{problem_signature}"

        if prob_id not in self.graph:
            self.graph.add_node(
                prob_id,
                data=ProblemNode(
                    signature=problem_signature,
                    description="",
                    domain="unknown",
                    hidden_structure="unknown",
                    constraints={},
                    best_algorithm="",
                    solved_count=0,
                    difficulty=0.5,
                    solution_history=[],
                ),
                node_type="problem",
            )

        if algo_id in self.graph:
            self.graph.add_edge(
                algo_id, prob_id,
                relation=EdgeType.FAILS_ON,
                reason=failure_reason,
                timestamp=str(time.time()),
            )
            algo = self.graph.nodes[algo_id]["data"]
            algo.times_used += 1

        self._save()

    def record_success(
        self,
        algorithm_name: str,
        problem_signature: str,
        quality: float,
    ) -> None:
        algo_id = f"algorithm:{algorithm_name}"
        prob_id = f"problem:{problem_signature}"

        if prob_id not in self.graph:
            self.graph.add_node(
                prob_id,
                data=ProblemNode(
                    signature=problem_signature,
                    description="",
                    domain="unknown",
                    hidden_structure="unknown",
                    constraints={},
                    best_algorithm=algorithm_name,
                    solved_count=0,
                    difficulty=0.5,
                    solution_history=[],
                ),
                node_type="problem",
            )

        existing = False
        for _, target, data in self.graph.out_edges(prob_id, data=True):
            if target == algo_id and data.get("relation") == EdgeType.SOLVED_BY:
                data["quality"] = max(data.get("quality", 0), quality)
                existing = True
                break

        if not existing:
            self.graph.add_edge(
                prob_id, algo_id,
                relation=EdgeType.SOLVED_BY,
                quality=quality,
            )

        if algo_id in self.graph:
            algo = self.graph.nodes[algo_id]["data"]
            algo.times_used += 1
            algo.times_succeeded += 1
            algo.performance_history.append(quality)
            if len(algo.performance_history) > 100:
                algo.performance_history = algo.performance_history[-100:]

        if prob_id in self.graph:
            prob = self.graph.nodes[prob_id]["data"]
            prob.solved_count += 1
            if quality > 0.8:
                prob.best_algorithm = algorithm_name

        self._save()

    def get_algorithm_code(self, name: str) -> str | None:
        algo_id = f"algorithm:{name}"
        if algo_id in self.graph:
            return self.graph.nodes[algo_id]["data"].code
        return None

    def get_algorithm(self, name: str) -> AlgorithmNode | None:
        algo_id = f"algorithm:{name}"
        if algo_id in self.graph:
            return self.graph.nodes[algo_id]["data"]
        return None

    def index_to_algorithm_name(self, idx: int) -> str:
        return self._index_algo.get(idx, f"algorithm_{idx}")

    def algorithm_name_to_index(self, name: str) -> int:
        return self._algo_index.get(name, 0)

    def stats(self) -> dict:
        n_algorithms = sum(
            1 for _, a in self.graph.nodes(data=True)
            if a.get("node_type") == "algorithm"
        )
        n_problems = sum(
            1 for _, a in self.graph.nodes(data=True)
            if a.get("node_type") == "problem"
        )
        n_principles = sum(
            1 for _, a in self.graph.nodes(data=True)
            if a.get("node_type") == "principle"
        )
        n_discovered = sum(
            1 for _, a in self.graph.nodes(data=True)
            if a.get("node_type") == "algorithm"
            and a["data"].discovered_by == "rl_synthesis"
        )
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "algorithms": n_algorithms,
            "problems": n_problems,
            "principles": n_principles,
            "discovered_algorithms": n_discovered,
            "algo_index_size": len(self._algo_index),
        }
