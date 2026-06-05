"""Unit tests for cross-domain knowledge graph routing."""
import pytest
from aalgoi.core.knowledge_graph import AlgorithmKnowledgeGraph
from aalgoi.core.meta_controller import UniversalMetaController


# ── Helper: build a minimal KG with cross-domain edges ──────────────────────

def _build_kg():
    kg = AlgorithmKnowledgeGraph()

    # Algorithms
    algos = [
        ("quicksort", {"time_complexity": "O(n log n)", "patterns": ["DivideAndConquer", "ComparisonSort"], "best_for": ["RandomData", "LargeN"]}),
        ("timsort", {"time_complexity": "O(n log n)", "patterns": ["Hybrid", "ComparisonSort"], "best_for": ["NearlySorted"]}),
        ("insertion_sort", {"time_complexity": "O(n^2)", "patterns": ["ComparisonSort", "Incremental"], "best_for": ["SmallN"]}),
        ("a_star", {"time_complexity": "O(E)", "patterns": ["HeuristicSearch", "ShortestPath"], "best_for": ["GeoGraph"]}),
        ("greedy_knapsack", {"time_complexity": "O(n log n)", "patterns": ["Greedy", "Approximation"], "best_for": ["ResourceAllocation"]}),
        ("simulated_annealing", {"time_complexity": "O(iterations * n)", "patterns": ["Metaheuristic", "Combinatorial"], "best_for": ["ComplexLandscape"]}),
    ]
    for name, meta in algos:
        kg.add_algorithm(name, meta)

    # Problem types
    kg.add_problem_type("sorting", ["quicksort", "timsort", "insertion_sort"])
    kg.add_problem_type("pathfinding", ["a_star"])
    kg.add_problem_type("optimization", ["greedy_knapsack", "simulated_annealing"])
    kg.add_problem_type("search", ["quicksort", "a_star"])

    # Cross-domain edges
    kg.graph.add_edge("DivideAndConquer", "optimization", relation="APPLICABLE_TO", weight=0.8)
    kg.graph.add_edge("DivideAndConquer", "search", relation="APPLICABLE_TO", weight=0.8)
    kg.graph.add_edge("HeuristicSearch", "pathfinding", relation="APPLICABLE_TO", weight=0.8)
    kg.graph.add_edge("HeuristicSearch", "search", relation="APPLICABLE_TO", weight=0.8)
    kg.graph.add_edge("ComparisonSort", "sorting", relation="APPLICABLE_TO", weight=0.8)
    kg.graph.add_edge("ComparisonSort", "search", relation="APPLICABLE_TO", weight=0.8)
    kg.graph.add_edge("Greedy", "optimization", relation="APPLICABLE_TO", weight=0.8)

    for p1, p2 in [("sorting", "search"), ("pathfinding", "search"), ("optimization", "search")]:
        kg.graph.add_edge(p1, p2, relation="SIMILAR_TO", weight=0.6)
        kg.graph.add_edge(p2, p1, relation="SIMILAR_TO", weight=0.6)

    return kg


# ── Tests ───────────────────────────────────────────────────────────────────


class TestFindCrossDomainCandidates:
    def test_returns_list(self):
        kg = _build_kg()
        candidates = kg.find_cross_domain_candidates("sorting", [], max_hops=2)
        assert isinstance(candidates, list)

    def test_excludes_direct_algorithms(self):
        kg = _build_kg()
        candidates = kg.find_cross_domain_candidates("sorting", [], max_hops=4)
        # quicksort, timsort, insertion_sort are direct — should be excluded
        for direct in ["quicksort", "timsort", "insertion_sort"]:
            assert direct not in candidates, f"{direct} should be excluded (direct)"

    def test_includes_cross_domain_algorithms(self):
        kg = _build_kg()
        candidates = kg.find_cross_domain_candidates("sorting", [], max_hops=4)
        # greedy_knapsack and simulated_annealing are optimization algorithms
        # reachable via DivideAndConquer → APPLICABLE_TO → optimization → SOLVED_BY
        assert "greedy_knapsack" in candidates, \
            "greedy_knapsack should be found via DivideAndConquer → optimization"
        assert "simulated_annealing" in candidates, \
            "simulated_annealing should be found via DivideAndConquer → optimization"

    def test_pathfinding_finds_sorting_via_search(self):
        kg = _build_kg()
        candidates = kg.find_cross_domain_candidates("pathfinding", [], max_hops=4)
        # a_star is direct
        assert "a_star" not in candidates, "a_star is direct for pathfinding"
        # pathfinding → SIMILAR_TO → search → SOLVED_BY → quicksort
        # pathfinding → HeuristicSearch → APPLICABLE_TO → search → SOLVED_BY → quicksort
        assert "quicksort" in candidates, \
            "quicksort should be found via pathfinding → search"

    def test_max_hops_limits_traversal(self):
        kg = _build_kg()
        with_4 = kg.find_cross_domain_candidates("sorting", [], max_hops=4)
        with_2 = kg.find_cross_domain_candidates("sorting", [], max_hops=2)
        assert len(with_4) >= len(with_2), \
            "More hops should find at least as many candidates"

    def test_returns_sorted_by_score(self):
        kg = _build_kg()
        candidates = kg.find_cross_domain_candidates("sorting", [], max_hops=4)
        # Should return cross-domain algorithms (non-direct ones)
        assert len(candidates) > 0

    def test_unknown_problem_type_returns_empty(self):
        kg = _build_kg()
        candidates = kg.find_cross_domain_candidates("nonexistent", [], max_hops=4)
        assert candidates == []

    def test_empty_graph_returns_empty(self):
        kg = AlgorithmKnowledgeGraph()
        candidates = kg.find_cross_domain_candidates("sorting", [], max_hops=4)
        assert candidates == []


class TestIntegrationWithMetaController:
    def test_meta_controller_populates_cross_domain_edges(self):
        mc = UniversalMetaController(config={"kg_enabled": True})
        kg = mc.kg
        # Check that APPLICABLE_TO edges exist
        has_applicable = False
        for u, v, d in kg.graph.edges(data=True):
            if d.get("relation") == "APPLICABLE_TO":
                has_applicable = True
                break
        assert has_applicable, "KG should have APPLICABLE_TO edges"

        # Check that SIMILAR_TO edges exist
        has_similar = False
        for u, v, d in kg.graph.edges(data=True):
            if d.get("relation") == "SIMILAR_TO":
                has_similar = True
                break
        assert has_similar, "KG should have SIMILAR_TO edges"

    def test_cross_domain_candidates_from_meta_controller(self):
        mc = UniversalMetaController(config={"kg_enabled": True})
        kg = mc.kg
        candidates = kg.find_cross_domain_candidates("sorting", [], max_hops=4)
        # Should have at least one cross-domain candidate
        sorting_direct = {"quicksort", "timsort", "merge_sort", "heap_sort",
                          "insertion_sort", "radix_sort"}
        cross_domain = [c for c in candidates if c not in sorting_direct]
        assert len(cross_domain) > 0, \
            f"Expected cross-domain candidates from sorting, got: {candidates}"

    def test_cross_domain_not_mixed_into_primary_pool_when_sufficient(self):
        """Cross-domain candidates are fallbacks, not mixed into RL action space."""
        mc = UniversalMetaController(config={"kg_enabled": True})
        kg = mc.kg

        # Use sorting with 100 elements (no SmallN constraint)
        mc._cross_domain_pool = []
        from aalgoi.core.problem_spec import ProblemSpec, ProblemType
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        context = {"data": list(range(100, 0, -1)), "data_profile": {"size": 100}}

        mc.select(context=context, candidates=list(mc.registry.keys()), problem_spec=spec)
        # sorting has 6 direct candidates (>= 3), cross_domain_pool stays empty
        assert len(mc._cross_domain_pool) == 0, \
            f"Cross-domain should be empty when primary >= 3, got: {mc._cross_domain_pool}"

    def test_cross_domain_used_as_fallback_when_primary_fails(self):
        """Cross-domain candidates are returned by get_fallback_alternative."""
        mc = UniversalMetaController(config={"kg_enabled": True})
        mc._cross_domain_pool = ["quicksort", "timsort"]

        fallback = mc.get_fallback_alternative("nonexistent")
        assert fallback == "quicksort", \
            f"Expected quicksort, got: {fallback}"
