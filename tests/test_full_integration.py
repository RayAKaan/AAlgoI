import pytest
import sys
import os
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem_spec import ProblemSpec, ProblemType
from pipeline import UniversalSolver


class TestSortingDomain:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_random_integers(self, solver):
        data = [random.randint(0, 1000) for _ in range(100)]
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, data)

        assert result['success'] is True
        assert len(result['result']) == len(data)
        assert all(result['result'][i] <= result['result'][i+1]
                   for i in range(len(result['result'])-1))

    def test_already_sorted(self, solver):
        data = list(range(1000))
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, data)

        assert result['success'] is True

    def test_reverse_sorted(self, solver):
        data = list(range(1000, 0, -1))
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, data)

        assert result['success'] is True

    def test_few_unique(self, solver):
        data = [random.choice([1, 5, 10, 100]) for _ in range(50)]
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, data)

        assert result['success'] is True


def generate_connected_graph(n, density=0.3):
    """Guarantees a path from start to end."""
    nodes = [str(i) for i in range(n)]
    graph = {node: {} for node in nodes}

    for i in range(n - 1):
        graph[nodes[i]][nodes[i + 1]] = random.randint(1, 10)

    for i in range(n):
        for j in range(i + 2, min(i + 5, n)):
            if random.random() < density:
                graph[nodes[i]][nodes[j]] = random.randint(1, 10)

    return {'graph': graph, 'start': nodes[0], 'end': nodes[-1]}


class TestPathfindingDomain:
    """Test pathfinding algorithm selection and execution."""

    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_sparse_graph(self, solver):
        data = generate_connected_graph(10, 0.4)
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        result = solver.solve(spec, data)

        assert result['success'] is True
        assert len(result['result']) > 0
        assert result['result'][0] == data['start']
        assert result['result'][-1] == data['end']

    def test_dense_graph(self, solver):
        data = generate_connected_graph(20, 0.6)
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        result = solver.solve(spec, data)

        assert result['success'] is True
        assert len(result['result']) > 0

    def test_no_path(self, solver):
        data = {'graph': {'A': {}, 'B': {}}, 'start': 'A', 'end': 'B'}
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        result = solver.solve(spec, data)

        assert result['success'] is False or len(result['result']) == 0


class TestOptimizationDomain:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def generate_knapsack(self, n):
        items = [{'value': random.randint(10, 200), 'weight': random.randint(1, 100)}
                 for _ in range(n)]
        capacity = sum(w['weight'] for w in items) // 2
        return {'items': items, 'capacity': capacity}

    def test_small_knapsack(self, solver):
        data = self.generate_knapsack(10)
        spec = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        result = solver.solve(spec, data)

        assert result['success'] is True
        assert 'selected' in result['result']
        assert 'value' in result['result']

    def test_medium_knapsack(self, solver):
        data = self.generate_knapsack(30)
        spec = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        result = solver.solve(spec, data)

        assert result['success'] is True
        assert len(result['result']['selected']) > 0

    def test_large_knapsack(self, solver):
        data = self.generate_knapsack(80)
        spec = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        result = solver.solve(spec, data)

        assert result['success'] is True


class TestRLIntegration:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_rl_agent_loaded(self, solver):
        assert solver.meta_controller.rl_agent is not None
        assert len(solver.meta_controller.rl_agent.network.state_dict()) > 0

    def test_knowledge_graph_loaded(self, solver):
        kg = solver.meta_controller.kg
        assert kg.graph.number_of_nodes() > 0
        assert kg.graph.number_of_edges() > 0
        assert kg.graph.has_node('sorting')
        assert kg.graph.has_node('quicksort')

    def test_fallback_mechanism(self, solver):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, "not_a_list")
        assert result['success'] is False


class TestResultContract:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_result_has_required_keys(self, solver):
        data = [3, 1, 2]
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, data)

        for key in ['result', 'algorithm', 'success', 'time_ms']:
            assert key in result, f"Missing key: {key}"

    def test_result_types(self, solver):
        data = [3, 1, 2]
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        result = solver.solve(spec, data)

        assert isinstance(result['result'], list)
        assert isinstance(result['algorithm'], str)
        assert isinstance(result['success'], bool)
        assert isinstance(result['time_ms'], (int, float))


class TestPerformance:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_sorting_speed(self, solver):
        data = [random.randint(0, 1000) for _ in range(1000)]
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)

        start = time.time()
        result = solver.solve(spec, data)
        elapsed = time.time() - start

        assert result['success'] is True
        assert elapsed < 5.0

    def test_pathfinding_speed(self, solver):
        nodes = [str(i) for i in range(50)]
        graph = {n: {} for n in nodes}
        for i in range(50):
            for j in range(i+1, min(i+3, 50)):
                if random.random() < 0.3:
                    graph[nodes[i]][nodes[j]] = random.randint(1, 10)

        data = {'graph': graph, 'start': '0', 'end': '49'}
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)

        start = time.time()
        result = solver.solve(spec, data)
        elapsed = time.time() - start

        assert result['success'] is True
        assert elapsed < 5.0
