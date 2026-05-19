import pytest
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem_spec import ProblemSpec, ProblemType
from pipeline import UniversalSolver


def generate_supply_chain_data(n_packages=50, n_cities=10, n_items=20):
    city_ids = [str(i) for i in range(n_cities)]
    graph = {}
    for i in range(n_cities):
        neighbors = {}
        for j in range(n_cities):
            if i != j and random.random() < 0.3:
                neighbors[str(j)] = random.randint(1, 100)
        graph[str(i)] = neighbors
    return {
        "packages": [random.randint(1, 100) for _ in range(n_packages)],
        "cities": {
            "graph": graph,
            "start": "0",
            "end": str(n_cities - 1)
        },
        "capacities": {
            "items": [
                {"value": random.randint(1, 100), "weight": random.randint(5, 50)}
                for _ in range(n_items)
            ],
            "capacity": 500
        }
    }


class TestMultiDomain:
    @pytest.fixture
    def solver(self):
        return UniversalSolver()

    def test_is_heterogeneous_dict(self, solver):
        data = generate_supply_chain_data()
        assert solver._is_heterogeneous_dict(data) is True

    def test_not_heterogeneous_simple_list(self, solver):
        assert solver._is_heterogeneous_dict([1, 2, 3]) is False

    def test_not_heterogeneous_small_dict(self, solver):
        assert solver._is_heterogeneous_dict({"a": 1, "b": 2}) is False

    def test_not_heterogeneous_uniform_types(self, solver):
        data = {"a": 1, "b": 2, "c": 3}
        assert solver._is_heterogeneous_dict(data) is False

    def test_auto_decompose_creates_sub_problems(self, solver):
        problem = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        data = generate_supply_chain_data()
        problem = solver._auto_decompose_problem(problem, data)
        assert len(problem.sub_problems) > 0
        assert "packages" in problem.sub_problems
        assert "cities" in problem.sub_problems
        assert "capacities" in problem.sub_problems

    def test_auto_decompose_sets_correct_types(self, solver):
        problem = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        data = generate_supply_chain_data()
        problem = solver._auto_decompose_problem(problem, data)
        assert problem.sub_problems["packages"].problem_type == ProblemType.SORTING
        assert problem.sub_problems["cities"].problem_type == ProblemType.PATHFINDING
        assert problem.sub_problems["capacities"].problem_type == ProblemType.OPTIMIZATION

    def test_auto_decompose_infers_pipeline_order(self, solver):
        problem = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        data = generate_supply_chain_data()
        problem = solver._auto_decompose_problem(problem, data)
        assert problem.pipeline_order == ["packages", "cities", "capacities"]

    def test_problem_spec_is_multi_domain_detection(self):
        sub = ProblemSpec(name="sub", problem_type=ProblemType.SORTING)
        spec = ProblemSpec(
            name="parent",
            problem_type=ProblemType.OPTIMIZATION,
            sub_problems={"sort": sub},
            pipeline_order=["sort"]
        )
        assert spec.is_multi_domain() is True

    def test_problem_spec_not_multi_domain_default(self):
        spec = ProblemSpec(name="simple", problem_type=ProblemType.SORTING)
        assert spec.is_multi_domain() is False

    def test_decompose_returns_sub_problems(self):
        sub = ProblemSpec(name="sub", problem_type=ProblemType.SORTING)
        spec = ProblemSpec(
            name="parent",
            problem_type=ProblemType.OPTIMIZATION,
            sub_problems={"sort": sub},
            pipeline_order=["sort"]
        )
        data = {"sort": [3, 1, 2]}
        result = spec.decompose(data)
        assert len(result) == 1
        assert result[0][0] is sub
        assert result[0][1] == [3, 1, 2]

    def test_multi_domain_supply_chain(self, solver):
        problem = ProblemSpec(
            name="supply_chain_test",
            problem_type=ProblemType.OPTIMIZATION
        )
        data = generate_supply_chain_data(20, 5)
        problem = solver._auto_decompose_problem(problem, data)
        result = solver.solve(problem, data)
        assert result['success'] is True
        assert isinstance(result['result'], dict)
        assert 'pipeline:' in result['algorithm']

    def test_auto_decompose_new_keys(self, solver):
        data = {
            "skus": [101, 102, 103],
            "disruptions": ["flood", "strike"],
            "bids": [10, 20, 30],
            "sequence": "ATCG",
            "matrix": [[1, 2], [3, 4]],
        }
        problem = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        problem = solver._auto_decompose_problem(problem, data)
        assert "skus" in problem.sub_problems
        assert "disruptions" in problem.sub_problems
        assert "bids" in problem.sub_problems
        assert "sequence" in problem.sub_problems
        assert "matrix" in problem.sub_problems
        assert problem.sub_problems["skus"].problem_type == ProblemType.SORTING
        assert problem.sub_problems["disruptions"].problem_type == ProblemType.CLASSIFICATION
        assert problem.sub_problems["bids"].problem_type == ProblemType.OPTIMIZATION
        assert problem.sub_problems["sequence"].problem_type == ProblemType.SEARCH
        assert problem.sub_problems["matrix"].problem_type == ProblemType.OPTIMIZATION

    def test_to_dict_roundtrip_with_sub_problems(self):
        sub = ProblemSpec(name="sub", problem_type=ProblemType.SORTING)
        spec = ProblemSpec(
            name="parent",
            problem_type=ProblemType.OPTIMIZATION,
            sub_problems={"sort": sub},
            pipeline_order=["sort"]
        )
        d = spec.to_dict()
        restored = ProblemSpec.from_dict(d)
        assert restored.name == "parent"
        assert "sort" in restored.sub_problems
        assert restored.sub_problems["sort"].problem_type == ProblemType.SORTING
        assert restored.pipeline_order == ["sort"]
