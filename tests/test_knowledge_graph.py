import pytest
from core.knowledge_graph import AlgorithmKnowledgeGraph


# ---------- Construction ----------

def test_add_algorithm_creates_nodes():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"patterns": ["DivideAndConquer"], "best_for": ["RandomData"]})
    assert kg.graph.has_node("quicksort")
    assert kg.graph.has_node("DivideAndConquer")
    assert kg.graph.has_node("RandomData")


def test_add_algorithm_creates_edges():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {
        "time_complexity": "O(n log n)",
        "patterns": ["DivideAndConquer"],
        "best_for": ["RandomData"],
    })
    assert kg.graph.has_edge("quicksort", "O(n log n)")
    assert kg.graph.has_edge("quicksort", "DivideAndConquer")
    assert kg.graph.has_edge("quicksort", "RandomData")


def test_add_algorithm_edge_relations():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"patterns": ["DivideAndConquer"]})
    edges = list(kg.graph.edges("quicksort", data=True))
    assert any(d["relation"] == "IS_A" for _, _, d in edges)


def test_add_problem_type_creates_problem_node():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {})
    kg.add_problem_type("sorting", ["quicksort"])
    assert kg.graph.has_node("sorting")
    assert kg.graph.nodes["sorting"]["type"] == "Problem"


def test_add_problem_type_links_to_algorithms():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {})
    kg.add_algorithm("timsort", {})
    kg.add_problem_type("sorting", ["quicksort", "timsort"])
    edges = list(kg.graph.edges("sorting", data=True))
    assert len(edges) == 2
    assert all(d["relation"] == "SOLVED_BY" for _, _, d in edges)


def test_safe_add_node_does_not_duplicate():
    kg = AlgorithmKnowledgeGraph()
    kg._safe_add_node("test", type="A")
    kg._safe_add_node("test", type="B")
    assert kg.graph.nodes["test"]["type"] == "A"


# ---------- find_candidates ----------

def test_find_candidates_returns_linked_algos():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {})
    kg.add_algorithm("timsort", {})
    kg.add_problem_type("sorting", ["quicksort", "timsort"])
    candidates = kg.find_candidates("sorting")
    assert "quicksort" in candidates
    assert "timsort" in candidates
    assert len(candidates) == 2


def test_find_candidates_unknown_problem_returns_empty():
    kg = AlgorithmKnowledgeGraph()
    assert kg.find_candidates("nonexistent") == []


def test_find_candidates_filters_by_constraint():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"best_for": ["RandomData"]})
    kg.add_algorithm("timsort", {"best_for": ["NearlySorted"]})
    kg.add_problem_type("sorting", ["quicksort", "timsort"])

    candidates = kg.find_candidates("sorting", constraints=["NearlySorted"])
    assert "timsort" in candidates
    assert "quicksort" not in candidates


def test_find_candidates_empty_constraint_falls_back():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"best_for": ["RandomData"]})
    kg.add_algorithm("timsort", {"best_for": ["NearlySorted"]})
    kg.add_problem_type("sorting", ["quicksort", "timsort"])

    candidates = kg.find_candidates("sorting", constraints=["NoMatch"])
    assert "quicksort" in candidates
    assert "timsort" in candidates


def test_find_candidates_no_constraints_returns_all():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {})
    kg.add_algorithm("timsort", {})
    kg.add_problem_type("sorting", ["quicksort", "timsort"])
    assert len(kg.find_candidates("sorting")) == 2


# ---------- find_alternatives ----------

def test_find_alternatives_returns_pattern_siblings():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"patterns": ["DivideAndConquer"]})
    kg.add_algorithm("merge_sort", {"patterns": ["DivideAndConquer"]})
    kg.add_algorithm("heap_sort", {"patterns": ["ComparisonSort"]})

    alternatives = kg.find_alternatives("quicksort")
    assert "merge_sort" in alternatives
    assert "heap_sort" not in alternatives


def test_find_alternatives_excludes_self():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"patterns": ["DivideAndConquer"]})
    kg.add_algorithm("merge_sort", {"patterns": ["DivideAndConquer"]})

    alternatives = kg.find_alternatives("quicksort")
    assert "quicksort" not in alternatives


def test_find_alternatives_unknown_algo_returns_empty():
    kg = AlgorithmKnowledgeGraph()
    assert kg.find_alternatives("nope") == []


def test_find_alternatives_no_patterns_returns_empty():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {})
    kg.add_algorithm("merge_sort", {"patterns": ["DivideAndConquer"]})
    assert kg.find_alternatives("quicksort") == []


def test_find_alternatives_across_shared_patterns():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("algo_a", {"patterns": ["P1", "P2"]})
    kg.add_algorithm("algo_b", {"patterns": ["P1"]})
    kg.add_algorithm("algo_c", {"patterns": ["P2"]})

    alts = kg.find_alternatives("algo_a")
    assert "algo_b" in alts
    assert "algo_c" in alts


# ---------- explain_path ----------

def test_explain_path_returns_string():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {})
    kg.add_problem_type("sorting", ["quicksort"])
    path = kg.explain_path("sorting", "quicksort")
    assert isinstance(path, str)
    assert "sorting" in path
    assert "quicksort" in path


def test_explain_path_no_path_returns_fallback():
    kg = AlgorithmKnowledgeGraph()
    path = kg.explain_path("nonexistent", "quicksort")
    assert path == "No semantic path found."


def test_explain_path_uses_shortest_path():
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm("quicksort", {"patterns": ["DivideAndConquer"]})
    kg.add_algorithm("merge_sort", {"patterns": ["DivideAndConquer"]})
    kg.add_problem_type("sorting", ["quicksort", "merge_sort"])

    path = kg.explain_path("sorting", "merge_sort")
    assert " → " in path
