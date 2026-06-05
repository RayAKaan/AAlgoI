
import pytest

from aalgoi.core.mind.knowledge_graph import (
    AlgorithmicKnowledgeGraph,
    AlgorithmNode,
    EdgeType,
)


@pytest.fixture
def kg(tmp_path):
    return AlgorithmicKnowledgeGraph(tmp_path / "kg_test")


class TestKGSeeding:
    def test_seeds_algorithms(self, kg):
        assert kg.stats()["algorithms"] == 85

    def test_seeds_principles(self, kg):
        assert kg.stats()["principles"] == 8

    def test_seeds_problems(self, kg):
        assert kg.stats()["problems"] == 16

    def test_algo_index_populated(self, kg):
        assert len(kg._algo_index) == 85
        assert "quick_sort" in kg._algo_index
        assert "dijkstra" in kg._algo_index
        assert "bfs" in kg._algo_index

    def test_index_round_trip(self, kg):
        idx = kg.algorithm_name_to_index("quick_sort")
        name = kg.index_to_algorithm_name(idx)
        assert name == "quick_sort"

    def test_edges_exist(self, kg):
        assert kg.stats()["total_edges"] > 0

    def test_algorithm_principle_edges(self, kg):
        algo_id = "algorithm:quick_sort"
        princ_id = "principle:divide_conquer"
        edges = list(kg.graph.out_edges(algo_id, data=True))
        principle_edges = [
            e for e in edges
            if e[1] == princ_id and e[2].get("relation") == EdgeType.USES_PRINCIPLE
        ]
        assert len(principle_edges) > 0

    def test_problem_algorithm_edges(self, kg):
        prob_id = "problem:SORTING"
        algo_id = "algorithm:tim_sort"
        edges = list(kg.graph.out_edges(prob_id, data=True))
        solved = [
            e for e in edges
            if e[1] == algo_id and e[2].get("relation") == EdgeType.SOLVED_BY
        ]
        assert len(solved) > 0

    def test_parent_algorithm_edges(self, kg):
        tim_id = "algorithm:tim_sort"
        edges = list(kg.graph.out_edges(tim_id, data=True))
        parents = [
            e[1] for e in edges
            if e[2].get("relation") == EdgeType.DERIVED_FROM
        ]
        assert "algorithm:merge_sort" in parents
        assert "algorithm:insertion_sort" in parents


class TestKGQueries:
    def test_query_similar_exact_match(self, kg):
        results = kg.query_similar_problems("SORTING")
        assert len(results) > 0
        assert results[0][1] == 1.0
        assert results[0][0].signature == "SORTING"

    def test_query_similar_keyword_match(self, kg):
        results = kg.query_similar_problems("sort array of integers")
        assert len(results) > 0
        domains = [r[0].domain for r in results]
        assert "integers" in domains

    def test_query_similar_graph_match(self, kg):
        results = kg.query_similar_problems("find path in graph with edges")
        assert len(results) > 0
        domains = [r[0].domain for r in results]
        assert "graph" in domains

    def test_get_best_algorithms_for_sorting(self, kg):
        algos = kg.get_best_algorithms_for("SORTING", {})
        assert len(algos) > 0
        names = [a.name for a in algos]
        assert any("sort" in n for n in names)

    def test_get_best_algorithms_for_pathfinding(self, kg):
        algos = kg.get_best_algorithms_for("PATHFINDING", {})
        assert len(algos) > 0
        names = [a.name for a in algos]
        assert "dijkstra" in names or "bfs" in names

    def test_get_known_failures_empty(self, kg):
        failures = kg.get_known_failures("SORTING")
        assert failures == []


class TestKGRecording:
    def test_record_failure(self, kg):
        kg.record_failure("bubble_sort", "SORTING", "too_slow")
        failures = kg.get_known_failures("SORTING")
        assert "bubble_sort" in failures

    def test_record_failure_updates_stats(self, kg):
        algo = kg.get_algorithm("bubble_sort")
        initial_used = algo.times_used
        kg.record_failure("bubble_sort", "SORTING", "too_slow")
        assert algo.times_used == initial_used + 1

    def test_record_success(self, kg):
        kg.record_success("quick_sort", "SORTING", 0.95)
        algo = kg.get_algorithm("quick_sort")
        assert algo.times_succeeded >= 1
        assert 0.95 in algo.performance_history

    def test_record_success_updates_best(self, kg):
        kg.record_success("quick_sort", "SORTING", 0.98)
        results = kg.query_similar_problems("SORTING")
        prob = results[0][0]
        assert prob.best_algorithm == "quick_sort"

    def test_record_new_algorithm(self, kg):
        from aalgoi.core.mind.cognitive_actions import CognitiveAction
        new_algo = AlgorithmNode(
            name="discovered_super_sort",
            code="def super_sort(arr): return sorted(arr)",
            time_complexity="O(n)", space_complexity="O(1)",
            principles=["optimal_substructure"],
            best_for=["sorting", "special_case"],
            discovered_by="rl_synthesis",
            performance_history=[0.99],
            correctness_verified=True,
            times_used=1, times_succeeded=1,
            created_at=str(1234567890),
            parent_algorithms=["quick_sort", "radix_sort"],
        )
        kg.record_new_algorithm(
            new_algo, "SORTING",
            [CognitiveAction.SYNTHESIZE_NEW, CognitiveAction.ACCEPT_SOLUTION],
        )

        assert kg.get_algorithm("discovered_super_sort") is not None
        assert kg.get_algorithm_code("discovered_super_sort") is not None

        idx = kg.algorithm_name_to_index("discovered_super_sort")
        assert kg.index_to_algorithm_name(idx) == "discovered_super_sort"

        assert kg.stats()["algorithms"] == 86
        assert kg.stats()["discovered_algorithms"] == 1

        algo_id = "algorithm:discovered_super_sort"
        edges = list(kg.graph.out_edges(algo_id, data=True))
        parent_edges = [
            e for e in edges
            if e[2].get("relation") == EdgeType.DERIVED_FROM
        ]
        assert len(parent_edges) == 2

    def test_record_new_algorithm_creates_problem(self, kg):
        new_algo = AlgorithmNode(
            name="novel_algo",
            code="def solve(x): return x",
            time_complexity="O(1)", space_complexity="O(1)",
            principles=[], best_for=["unknown"],
            discovered_by="rl_synthesis",
            performance_history=[0.5],
            correctness_verified=True,
            times_used=1, times_succeeded=1,
            created_at="0", parent_algorithms=[],
        )
        kg.record_new_algorithm(new_algo, "TOTALLY_NEW_PROBLEM", [])
        results = kg.query_similar_problems("TOTALLY_NEW_PROBLEM")
        assert len(results) > 0
        assert results[0][1] == 1.0


class TestKGPersistence:
    def test_persists_and_loads(self, tmp_path):
        kg_path = tmp_path / "kg_persist"
        kg1 = AlgorithmicKnowledgeGraph(kg_path)
        kg1.record_success("quick_sort", "SORTING", 0.9)
        stats1 = kg1.stats()

        kg2 = AlgorithmicKnowledgeGraph(kg_path)
        stats2 = kg2.stats()

        assert stats1["algorithms"] == stats2["algorithms"]
        assert stats1["total_edges"] == stats2["total_edges"]
        algo = kg2.get_algorithm("quick_sort")
        assert algo.times_succeeded >= 1

    def test_failure_persists(self, tmp_path):
        kg_path = tmp_path / "kg_persist2"
        kg1 = AlgorithmicKnowledgeGraph(kg_path)
        kg1.record_failure("bubble_sort", "SORTING", "slow")

        kg2 = AlgorithmicKnowledgeGraph(kg_path)
        failures = kg2.get_known_failures("SORTING")
        assert "bubble_sort" in failures


class TestKGLookups:
    def test_get_algorithm_code(self, kg):
        code = kg.get_algorithm_code("quick_sort")
        assert code is not None
        assert "quick_sort" in code

    def test_get_algorithm_code_unknown(self, kg):
        code = kg.get_algorithm_code("nonexistent_algo")
        assert code is None

    def test_get_algorithm(self, kg):
        algo = kg.get_algorithm("dijkstra")
        assert algo is not None
        assert algo.time_complexity == "O((V+E) log V)"

    def test_get_algorithm_unknown(self, kg):
        algo = kg.get_algorithm("nonexistent")
        assert algo is None

    def test_index_to_algorithm_name_invalid(self, kg):
        name = kg.index_to_algorithm_name(9999)
        assert "algorithm_9999" in name


class TestKGStats:
    def test_stats_structure(self, kg):
        stats = kg.stats()
        for key in ("total_nodes", "total_edges", "algorithms",
                     "problems", "principles", "discovered_algorithms",
                     "algo_index_size"):
            assert key in stats

    def test_stats_values_reasonable(self, kg):
        stats = kg.stats()
        assert stats["algorithms"] == 85
        assert stats["principles"] == 8
        assert stats["problems"] == 16
        assert stats["discovered_algorithms"] == 0
        assert stats["total_nodes"] == 85 + 8 + 16
        assert stats["total_edges"] > 100
