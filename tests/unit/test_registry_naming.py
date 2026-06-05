import pytest
import re


def test_name_must_be_snake_case():
    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()
    for name, algo in solver.registry.items():
        assert re.match(r'^[a-z][a-z0-9_]*$', name), (
            f"Algorithm '{name}' does not follow snake_case naming convention"
        )


def test_no_duplicate_names():
    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()
    names = list(solver.registry.keys())
    assert len(names) == len(set(names)), "Duplicate algorithm names found"


def test_validate_name_rejects_bad_names():
    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()
    with pytest.raises(ValueError, match="must be snake_case"):
        solver._validate_name("CamelCase")
    with pytest.raises(ValueError, match="must be snake_case"):
        solver._validate_name("has spaces")
    with pytest.raises(ValueError, match="must be snake_case"):
        solver._validate_name("starts_with_Upper")
    solver._validate_name("valid_name_123")


def test_register_algorithm_rejects_duplicate():
    from aalgoi.algorithms.base import Algorithm

    class MockAlgo(Algorithm):
        name = "mock_algo"
        def process(self, data):
            return data

    class MockAlgo2(Algorithm):
        name = "mock_algo"
        def process(self, data):
            return data

    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()
    solver.register_algorithm(MockAlgo())
    with pytest.raises(KeyError, match="Duplicate algorithm registration"):
        solver.register_algorithm(MockAlgo2())
