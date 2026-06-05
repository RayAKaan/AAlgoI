"""Integration tests verifying pipeline end-to-end for supported problem types."""

import numpy as np
import pytest

from aalgoi.core.problem_spec import ProblemSpec, ProblemType
from aalgoi.pipeline import UniversalSolver


@pytest.fixture
def solver():
    return UniversalSolver()


def test_sorting_solve(solver):
    spec = ProblemSpec(name="sort_test", problem_type=ProblemType.SORTING)
    result = solver.solve(spec, [3, 1, 4, 1, 5, 9, 2, 6])
    assert result["success"]
    assert result["result"] == sorted([3, 1, 4, 1, 5, 9, 2, 6])


def test_sorting_empty(solver):
    spec = ProblemSpec(name="sort_empty", problem_type=ProblemType.SORTING)
    result = solver.solve(spec, [])
    assert result["success"]
    assert result["result"] == []


def test_oracle_gates_zero_reward_on_failure(solver):
    from aalgoi.core.oracles import evaluate
    result = evaluate(ProblemType.SORTING, [3, 1, 2], [3, 2, 1])
    assert result is False


def test_image_processing_must_preserve_shape(solver):
    spec = ProblemSpec(name="img_test", problem_type=ProblemType.IMAGE_PROCESSING)
    data = np.random.randn(10, 10).astype(np.float32)
    result = solver.solve(spec, data)
    assert result["success"]
    processed = result.get("result", {})
    if isinstance(processed, dict) and "result" in processed:
        processed = processed["result"]
    processed_arr = np.asarray(processed)
    assert processed_arr.shape == data.shape, (
        f"Expected shape {data.shape}, got {processed_arr.shape}"
    )


def test_ml_validation_rejects_1d_input(solver):
    from aalgoi.algorithms.ml.base import MLAlgorithm

    class MockML(MLAlgorithm):
        def __init__(self):
            super().__init__(model_class=None, name="mock_ml")

    algo = MockML()
    result = algo._validate_ml_input([1, 2, 3])
    assert result is not None
    assert result.ndim == 2


def test_question_parser_returns_unknown_for_ambiguous():
    from aalgoi.core.question_parser import QuestionParser
    parser = QuestionParser()
    result = parser.parse("do something with data")
    assert result.problem_type == ProblemType.UNKNOWN


def test_knowledge_graph_compatibility_oracle():
    from aalgoi.core.knowledge_graph import AlgorithmKnowledgeGraph
    kg = AlgorithmKnowledgeGraph()
    assert kg._compatibility_oracle("SORTING", "SEARCH") is True
    assert kg._compatibility_oracle("SORTING", "NLP") is False
