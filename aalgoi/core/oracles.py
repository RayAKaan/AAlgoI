from collections.abc import Callable
from typing import Any

import numpy as np

from aalgoi.core.problem_spec import ProblemType

ORACLES: dict[ProblemType, Callable[[Any, Any], bool]] = {}

def _register(pt: ProblemType):
    def decorator(fn: Callable[[Any, Any], bool]):
        ORACLES[pt] = fn
        return fn
    return decorator


def get_oracle(problem_type: ProblemType) -> Callable[[Any, Any], bool] | None:
    return ORACLES.get(problem_type)


def evaluate(problem_type: ProblemType, input_data: Any, output_data: Any) -> bool:
    oracle_fn = get_oracle(problem_type)
    if oracle_fn is None:
        return True
    return oracle_fn(input_data, output_data)


@_register(ProblemType.SORTING)
def _sorting_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(inp, np.ndarray) and isinstance(out, np.ndarray):
        return np.array_equal(np.sort(inp), out)
    if isinstance(inp, (list, tuple)) and isinstance(out, (list, tuple)):
        if len(inp) != len(out):
            return False
        return list(out) == sorted(inp)
    return False


@_register(ProblemType.PATHFINDING)
def _pathfinding_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(out, (list, tuple)):
        return len(out) > 0
    return True


@_register(ProblemType.SEARCH)
def _search_oracle(inp: Any, out: Any) -> bool:
    return out is not None


@_register(ProblemType.CLASSIFICATION)
def _classification_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(out, dict):
        preds = out.get("predictions")
        if preds is not None:
            return len(preds) > 0
        return out.get("trained", False)
    return False


@_register(ProblemType.REGRESSION)
def _regression_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(out, dict):
        preds = out.get("predictions")
        if preds is not None:
            return len(preds) > 0
        return out.get("trained", False)
    return False


@_register(ProblemType.CLUSTERING)
def _clustering_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(out, dict):
        labels = out.get("labels") or out.get("predictions")
        if labels is not None and hasattr(labels, '__len__'):
            inp_len = len(inp) if hasattr(inp, '__len__') else 0
            return len(labels) == inp_len if inp_len > 0 else len(labels) > 0
        return out.get("trained", False)
    return False


@_register(ProblemType.OPTIMIZATION)
def _optimization_oracle(inp: Any, out: Any) -> bool:
    return out is not None


@_register(ProblemType.SCHEDULING)
def _scheduling_oracle(inp: Any, out: Any) -> bool:
    return out is not None and (not hasattr(out, '__len__') or len(out) > 0)


@_register(ProblemType.ROUTING)
def _routing_oracle(inp: Any, out: Any) -> bool:
    return out is not None and (not hasattr(out, '__len__') or len(out) > 0)


@_register(ProblemType.NLP)
def _nlp_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(out, (list, tuple)):
        return len(out) > 0
    if isinstance(out, str):
        return len(out.strip()) > 0
    return True


@_register(ProblemType.IMAGE_PROCESSING)
def _image_processing_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(inp, np.ndarray) and isinstance(out, np.ndarray):
        return inp.shape == out.shape
    return True


@_register(ProblemType.COMPUTER_VISION)
def _computer_vision_oracle(inp: Any, out: Any) -> bool:
    return out is not None


@_register(ProblemType.TRANSFORMATION)
def _transformation_oracle(inp: Any, out: Any) -> bool:
    return out is not None


@_register(ProblemType.GENERATION)
def _generation_oracle(inp: Any, out: Any) -> bool:
    return out is not None and (not isinstance(out, str) or len(out.strip()) > 0)


@_register(ProblemType.DECISION)
def _decision_oracle(inp: Any, out: Any) -> bool:
    if isinstance(out, bool):
        return True
    if isinstance(out, dict):
        decision = out.get("decision") or out.get("result")
        return decision is not None
    return out is not None


@_register(ProblemType.ML)
def _ml_oracle(inp: Any, out: Any) -> bool:
    if out is None:
        return False
    if isinstance(out, dict):
        return out.get("trained", False)
    return False
