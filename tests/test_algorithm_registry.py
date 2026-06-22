"""Tests for the algorithm registry and registration system."""
import pytest
import tempfile
import os

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.sorting import QuickSort, TimSort, MergeSort, HeapSort, InsertionSort, RadixSort
from aalgoi.algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder
from aalgoi.algorithms.optimization import GreedyKnapsack, SimulatedAnnealing, GeneticAlgorithm, HillClimbing, ParticleSwarm, AntColony
from aalgoi.algorithms.safety import IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack
from aalgoi.algorithms.ml import KMeansClustering, DBSCANClustering, RandomForestClassifier, LinearRegression
from aalgoi.core.knowledge_graph import AlgorithmKnowledgeGraph


class TestAlgorithmInheritance:
    def test_all_optimization_inherit_from_algorithm(self):
        for cls in [GreedyKnapsack, SimulatedAnnealing, GeneticAlgorithm, HillClimbing, ParticleSwarm, AntColony]:
            obj = cls()
            assert isinstance(obj, Algorithm)
            assert hasattr(obj, 'name')
            assert obj.name
            assert hasattr(obj, 'time_complexity')
            assert hasattr(obj, 'space_complexity')
            assert hasattr(obj, 'tags')
            assert hasattr(obj, 'best_for')
            assert hasattr(obj, 'patterns')
            assert hasattr(obj, 'problem_types')

    def test_all_sorting_inherit_from_algorithm(self):
        for cls in [QuickSort, TimSort, MergeSort, HeapSort, InsertionSort, RadixSort]:
            obj = cls()
            assert isinstance(obj, Algorithm)
            assert hasattr(obj, 'name') and obj.name
            assert hasattr(obj, 'process')

    def test_all_pathfinding_inherit_from_algorithm(self):
        for cls in [Dijkstra, AStar, BFSPathfinder]:
            obj = cls()
            assert isinstance(obj, Algorithm)
            assert hasattr(obj, 'name') and obj.name
            assert hasattr(obj, 'process')

    def test_all_safety_inherit_from_algorithm(self):
        for cls in [IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack]:
            obj = cls()
            assert isinstance(obj, Algorithm)
            assert hasattr(obj, 'name') and obj.name
            assert hasattr(obj, 'process')

    def test_all_ml_inherit_from_algorithm(self):
        for cls in [KMeansClustering, DBSCANClustering, RandomForestClassifier, LinearRegression]:
            obj = cls()
            assert isinstance(obj, Algorithm)
            assert hasattr(obj, 'name') and obj.name


class TestAlgorithmMetadata:
    def test_greedy_knapsack_metadata(self):
        algo = GreedyKnapsack()
        assert algo.name == "greedy_knapsack"
        assert "optimization" in algo.tags
        assert "OPTIMIZATION" in algo.problem_types

    def test_tim_sort_metadata(self):
        algo = TimSort()
        assert "tim" in algo.name.lower()
        assert "SORTING" in algo.problem_types

    def test_dijkstra_metadata(self):
        algo = Dijkstra()
        assert "dijkstra" in algo.name.lower()
        assert "PATHFINDING" in algo.problem_types

    def test_safe_sort_metadata(self):
        algo = SafeSort()
        assert algo.name == "safe_sort"
        assert "safety" in algo.tags or "fallback" in algo.tags

    def test_identity_metadata(self):
        algo = IdentityAlgorithm()
        assert algo.name == "identity"
        assert "safety" in algo.tags


class TestAlgorithmProcess:
    def test_identity_process(self):
        algo = IdentityAlgorithm()
        data = [3, 1, 2]
        result = algo.process(data)
        assert result == data

    def test_safe_sort_process(self):
        algo = SafeSort()
        result = algo.process([3, 1, 4, 1, 5])
        assert result == [1, 1, 3, 4, 5]

    def test_safe_sort_process_with_unsortable(self):
        algo = SafeSort()
        result = algo.process("not a list")
        assert result == sorted("not a list")

    def test_safe_path_process(self):
        algo = SafePath()
        data = {
            "graph": {"A": {"B": 1}, "B": {}},
            "start": "A",
            "end": "B"
        }
        result = algo.process(data)
        assert result == ["A", "B"]

    def test_safe_path_process_invalid(self):
        algo = SafePath()
        result = algo.process({})
        assert result == []

    def test_safe_knapsack_process(self):
        algo = SafeKnapsack()
        data = {
            "items": [{"value": 3, "weight": 2}],
            "capacity": 5
        }
        result = algo.process(data)
        assert "selected" in result
        assert "value" in result
        assert "weight" in result

    def test_safe_knapsack_process_invalid(self):
        algo = SafeKnapsack()
        result = algo.process({})
        assert result == {"selected": [], "value": 0, "weight": 0}


class TestAlgorithmCloning:
    def test_clone_identity(self):
        algo = IdentityAlgorithm()
        clone = algo.clone()
        assert clone is not algo
        assert clone.name == algo.name
        assert clone.tags == algo.tags

    def test_clone_safe_sort(self):
        algo = SafeSort()
        clone = algo.clone()
        assert clone is not algo
        assert clone.name == algo.name

    def test_clone_modifies_params(self):
        algo = SafeSort()
        algo.params["reverse"] = True
        clone = algo.clone()
        clone.params["reverse"] = False
        assert algo.params["reverse"] is True


class TestKnowledgeGraph:
    def test_kg_add_algorithm(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("test_algo", {
            "time_complexity": "O(n)",
            "patterns": ["test"],
            "best_for": ["testing"]
        })
        assert kg.graph.has_node("test_algo")
        assert kg.graph.nodes["test_algo"]["type"] == "Algorithm"

    def test_kg_add_problem_type(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("sort", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_problem_type("SORTING", ["sort"])
        assert kg.graph.has_node("SORTING")
        assert kg.graph.has_node("sort")

    def test_kg_find_candidates(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("sort", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_problem_type("SORTING", ["sort"])
        candidates = kg.find_candidates("SORTING")
        assert "sort" in candidates

    def test_kg_find_candidates_no_constraints(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("sort", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_algorithm("quick", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_problem_type("SORTING", ["sort", "quick"])
        candidates = kg.find_candidates("SORTING")
        assert len(candidates) == 2

    def test_kg_find_alternatives(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("sort", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_algorithm("quick", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        alternatives = kg.find_alternatives("sort")
        assert "quick" in alternatives

    def test_kg_find_alternatives_no_node(self):
        kg = AlgorithmKnowledgeGraph()
        alternatives = kg.find_alternatives("nonexistent")
        assert alternatives == []

    def test_kg_explain_path(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("sort", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_problem_type("SORTING", ["sort"])
        path = kg.explain_path("SORTING", "sort")
        assert "SORTING" in path
        assert "sort" in path
        assert "\u2192" in path

    def test_kg_explain_path_no_path(self):
        kg = AlgorithmKnowledgeGraph()
        path = kg.explain_path("A", "B")
        assert "No semantic path found" in path

    def test_kg_cross_domain(self):
        kg = AlgorithmKnowledgeGraph()
        kg.add_algorithm("sort", {"time_complexity": "O(n log n)", "patterns": ["divide_conquer"], "best_for": ["sorting"]})
        kg.add_problem_type("SORTING", ["sort"])
        kg.add_problem_type("SEARCH", [])
        kg.add_cross_domain_edge("SORTING", "SEARCH", "SIMILAR_TO", 0.6)
        candidates = kg.find_cross_domain_candidates("SORTING", [])
        assert isinstance(candidates, list)
