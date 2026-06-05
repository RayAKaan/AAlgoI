import numpy as np

from aalgoi.core.oracles import ORACLES, evaluate, get_oracle
from aalgoi.core.problem_spec import ProblemType


def test_all_16_oracles_registered():
    expected_count = len([pt for pt in ProblemType if pt != ProblemType.UNKNOWN])
    assert len(ORACLES) == expected_count, (
        f"Expected {expected_count} oracles, got {len(ORACLES)}"
    )


def test_sorting_oracle():
    oracle = get_oracle(ProblemType.SORTING)
    assert oracle is not None
    assert oracle([3, 1, 2], [1, 2, 3]) is True
    assert oracle([3, 1, 2], [3, 2, 1]) is False
    assert oracle([], []) is True
    assert oracle(np.array([3, 1, 2]), np.array([1, 2, 3])) is True
    assert oracle(None, None) is False


def test_pathfinding_oracle():
    oracle = get_oracle(ProblemType.PATHFINDING)
    assert oracle is not None
    assert oracle({}, [1, 2, 3]) is True
    assert oracle({}, []) is False
    assert oracle({}, None) is False


def test_classification_oracle():
    oracle = get_oracle(ProblemType.CLASSIFICATION)
    assert oracle is not None
    assert oracle({}, {"predictions": [0, 1, 0]}) is True
    assert oracle({}, {"trained": True}) is True
    assert oracle({}, {"trained": False}) is False
    assert oracle({}, None) is False


def test_clustering_oracle():
    oracle = get_oracle(ProblemType.CLUSTERING)
    assert oracle is not None
    assert oracle([1, 2, 3], {"labels": [0, 0, 1]}) is True
    assert oracle([1, 2, 3], {}) is False


def test_ml_oracle():
    oracle = get_oracle(ProblemType.ML)
    assert oracle is not None
    assert oracle({}, {"trained": True}) is True
    assert oracle({}, {"trained": False}) is False
    assert oracle({}, None) is False


def test_image_processing_oracle():
    oracle = get_oracle(ProblemType.IMAGE_PROCESSING)
    assert oracle is not None
    inp = np.zeros((10, 10))
    out = np.zeros((10, 10))
    assert oracle(inp, out) is True
    wrong = np.zeros((5, 5))
    assert oracle(inp, wrong) is False
    assert oracle(None, None) is False


def test_unknown_pt_returns_true():
    assert evaluate(ProblemType.UNKNOWN, {}, {}) is True


def test_evaluate_integration():
    assert evaluate(ProblemType.SORTING, [3, 1, 2], [1, 2, 3]) is True
    assert evaluate(ProblemType.SORTING, [3, 1, 2], [3, 2, 1]) is False
