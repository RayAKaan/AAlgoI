"""
Property-based tests for algorithm correctness using Hypothesis.

Ensures every algorithm satisfies domain-specific correctness
properties across a wide range of random inputs.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from aalgoi.core.oracles import evaluate
from aalgoi.core.problem_spec import ProblemType


@settings(max_examples=10, deadline=None)
@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=50))
def test_sorting_property(data):
    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()
    from aalgoi.core.problem_spec import ProblemSpec
    spec = ProblemSpec(name="sort_test", problem_type=ProblemType.SORTING)
    result = solver.solve(spec, data)
    assert result.get("success", False), f"Sorting failed on {data}"
    output = result.get("result", [])
    assert evaluate(ProblemType.SORTING, data, output), (
        f"Sorting oracle rejected: {data} -> {output}"
    )


@settings(max_examples=30, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=30,
    ),
)
def test_ml_validate_input_property(data):
    from aalgoi.algorithms.ml.base import MLAlgorithm

    class MockAlgo(MLAlgorithm):
        def __init__(self):
            super().__init__(model_class=None, name="mock")

    algo = MockAlgo()
    result = algo._validate_ml_input(data)
    assert result is not None
    assert result.ndim == 2
    assert result.shape[0] == len(data)
    assert result.shape[1] == 1
