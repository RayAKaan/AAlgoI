"""Stress test: LLM + sandbox algorithm synthesis."""
from unittest.mock import patch

from aalgoi.core.algorithm_synthesizer import LLMAlgorithmSynthesizer
from aalgoi.core.llm_client import OllamaClient
from aalgoi.core.problem_spec import ProblemSpec, ProblemType


def test_synthesis_never_crashes_solver():
    """Synthesizer failure never propagates to solver."""
    from aalgoi.core.smart_solver import SmartSolver
    solver = SmartSolver()
    result = solver.ask("sort this list", [5, 3, 1, 4, 2])
    assert result.get("success", False)
    assert result.get("algorithm") in (
        "timsort", "quicksort", "insertion_sort",
        "merge_sort", "heap_sort", "radix_sort",
    )


def test_synthesis_sandbox_rejects_malicious_code():
    """Sandbox blocks imports and eval even through synthesizer."""
    with patch.object(OllamaClient, '_test_connection', return_value=None):
        with patch.object(OllamaClient, 'generate') as mock_gen:
            mock_gen.return_value = """
import os
def process(data):
    os.system('echo hacked')
    return data
"""
            synth = LLMAlgorithmSynthesizer()
            spec = ProblemSpec("test", ProblemType.SORTING, [], [])
            algo = synth.synthesize(spec, list(range(10, 30)))
            assert algo is None, "Malicious code must be rejected"


def test_synthesis_empty_code_fallback():
    """Empty LLM output falls back gracefully with None."""
    with patch.object(OllamaClient, '_test_connection', return_value=None):
        with patch.object(OllamaClient, 'generate') as mock_gen:
            mock_gen.return_value = ""
            synth = LLMAlgorithmSynthesizer()
            spec = ProblemSpec("test", ProblemType.SORTING, [], [])
            algo = synth.synthesize(spec, list(range(10, 30)))
            assert algo is None
