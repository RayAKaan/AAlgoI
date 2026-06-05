"""Tests for LLM + sandbox algorithm synthesis."""
import sys
import ast
from unittest.mock import MagicMock, patch

import pytest

from aalgoi.core.algorithm_synthesizer import LLMAlgorithmSynthesizer
from aalgoi.core.llm_client import OllamaClient
from aalgoi.core.sandboxed_executor import (
    create_sandboxed_module, execute_sandboxed, benchmark_sandboxed,
)
from aalgoi.core.problem_spec import ProblemSpec, ProblemType


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama():
    with patch.object(OllamaClient, '_test_connection', return_value=None):
        with patch.object(OllamaClient, 'generate') as mock_gen:
            mock_gen.return_value = """
def process(data):
    return sorted(data)[:5]
"""
            yield mock_gen


@pytest.fixture
def synth(mock_ollama):
    return LLMAlgorithmSynthesizer()


@pytest.fixture
def sort_spec():
    return ProblemSpec(
        name="test_sort",
        problem_type=ProblemType.SORTING,
        constraints=[],
        objectives=[],
    )


# ── LLM Client Tests ─────────────────────────────────────────────────────

def test_ollama_connection_refused():
    """Ollama client raises RuntimeError if Ollama is unreachable."""
    with patch('aalgoi.core.llm_client.requests.get') as mock_get:
        mock_get.side_effect = ConnectionError("refused")
        with pytest.raises(RuntimeError, match="Ollama not found"):
            OllamaClient(base_url="http://localhost:19999")


def test_ollama_generate_failure():
    """Ollama client raises RuntimeError on generation failure."""
    with patch.object(OllamaClient, '_test_connection', return_value=None):
        with patch('aalgoi.core.llm_client.requests.post') as mock_post:
            mock_post.side_effect = RuntimeError("API error")
            client = OllamaClient()
            with pytest.raises(RuntimeError, match="Ollama generation failed"):
                client.generate("test prompt")


def test_ollama_generate_success():
    """Ollama client returns generated text."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "def process(data): pass"}

    with patch.object(OllamaClient, '_test_connection', return_value=None):
        with patch('aalgoi.core.llm_client.requests.post') as mock_post:
            mock_post.return_value = mock_resp
            client = OllamaClient()
            code = client.generate("write code")
            assert "def process(data): pass" in code


def test_ollama_model_available():
    """is_model_available checks model list."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "llama2"}, {"name": "mistral"}]
    }

    with patch.object(OllamaClient, '_test_connection', return_value=None):
        with patch('aalgoi.core.llm_client.requests.get') as mock_get:
            mock_get.return_value = mock_resp
            client = OllamaClient()
            assert client.is_model_available("llama2") is True
            assert client.is_model_available("codellama") is False


# ── Synthesizer Tests ────────────────────────────────────────────────────

def test_synthesize_basic(synth, sort_spec):
    """Synthesizer returns an Algorithm when LLM generates valid code."""
    data = list(range(10, 20))
    algo = synth.synthesize(sort_spec, data, baseline_algo=None)
    assert algo is not None
    assert algo.name.startswith("synth_")
    assert "sorting" in algo.tags
    result = algo.process(data)
    assert result == [10, 11, 12, 13, 14]


def test_synthesize_no_def_process(synth, mock_ollama, sort_spec):
    """Synthesizer returns None when LLM code lacks process()."""
    mock_ollama.return_value = "def foo(data): pass"
    algo = synth.synthesize(sort_spec, [1, 2, 3])
    assert algo is None


def test_synthesize_empty_code(synth, mock_ollama, sort_spec):
    """Synthesizer returns None on empty code."""
    mock_ollama.return_value = ""
    algo = synth.synthesize(sort_spec, [1, 2, 3])
    assert algo is None


def test_synthesize_llm_fails(synth, mock_ollama, sort_spec):
    """Synthesizer returns None when LLM raises."""
    mock_ollama.side_effect = RuntimeError("Ollama down")
    algo = synth.synthesize(sort_spec, [1, 2, 3])
    assert algo is None


def test_synthesize_skips_small_data(synth, sort_spec):
    """Synthesizer skips on data < 10 elements."""
    algo = synth.synthesize(sort_spec, [1, 2, 3])
    assert algo is None


def test_synthesize_skips_large_data(synth, sort_spec):
    """Synthesizer skips on data > 1000 elements."""
    algo = synth.synthesize(sort_spec, list(range(1001)))
    assert algo is None


def test_synthesize_validates_output(synth, mock_ollama, sort_spec):
    """Synthesizer validates that process() returns a result."""
    mock_ollama.return_value = """
def process(data):
    return [x * 2 for x in data]
"""
    data = list(range(10, 20))
    algo = synth.synthesize(sort_spec, data)
    assert algo is not None
    result = algo.process(data)
    assert result == [x * 2 for x in data]


def test_synthesize_benchmark_rejects_slow(synth, mock_ollama, sort_spec):
    """Synthesizer returns None if synthesized algo is slower than baseline."""
    mock_ollama.return_value = """
def process(data):
    # intentionally slow O(n^2)
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                result.append(data[i])
    return result
"""
    data = list(range(10, 110))  # Larger to make timing difference clear

    class FastBaseline:
        def process(self, data):
            return list(data)
        name = "fast"

    algo = synth.synthesize(sort_spec, data, baseline_algo=FastBaseline())
    # The O(n^2) algo should be slower than baseline O(n)
    assert algo is None


def test_synthesize_registers_in_kg(synth, mock_ollama, sort_spec):
    """Synthesized algorithm can be added to KG."""
    data = list(range(10, 20))
    mock_ollama.return_value = """
def process(data):
    return sorted(data)
"""
    algo = synth.synthesize(sort_spec, data)
    assert algo is not None

    from aalgoi.core.knowledge_graph import AlgorithmKnowledgeGraph
    kg = AlgorithmKnowledgeGraph()
    kg.add_algorithm(algo.name, {
        "time_complexity": algo.time_complexity,
        "best_for": algo.best_for,
    })
    kg.add_problem_type("sorting", [algo.name])

    cats = kg.find_candidates("sorting", [])
    assert algo.name in cats


# ── Edge Case Tests ──────────────────────────────────────────────────────

def test_sandbox_rejects_syntax_error():
    """Sandbox rejects malformed code."""
    code = "def process(data): return data +"
    assert create_sandboxed_module("test", code) is None


def test_sandbox_rejects_runtime_error():
    """Sandbox catches runtime errors gracefully."""
    code = """
def process(data):
    return data.nonexistent
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", [1])
    assert not success


def test_empty_list_input():
    """Sandbox handles empty list input."""
    code = """
def process(data):
    return sorted(data)
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", [])
    assert success
    assert result == []


def test_sandbox_string_input():
    """Sandbox handles string input."""
    code = """
def process(data):
    return data.upper()
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", "hello")
    assert success
    assert result == "HELLO"


def test_sandbox_int_input():
    """Sandbox handles integer input."""
    code = """
def process(data):
    return data * 2
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", 21)
    assert success
    assert result == 42


def test_benchmark_returns_times():
    """benchmark_sandboxed returns timing data."""
    code = """
def process(data):
    return sorted(data)
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    data = list(range(10, 30))
    baseline = lambda d: sorted(d)
    ok, synth_time, base_time = benchmark_sandboxed(
        module, "process", data, baseline, trials=3
    )
    assert ok
    assert synth_time > 0
    assert base_time > 0


# ── Complexity Inference Tests ──────────────────────────────────────────

def test_infer_constant():
    tree = ast.parse("def process(d): return d")
    c = LLMAlgorithmSynthesizer._infer_complexity(tree)
    assert c['time'] == 'O(1)'


def test_infer_linear():
    tree = ast.parse("def process(d):\n    for x in d:\n        print(x)")
    c = LLMAlgorithmSynthesizer._infer_complexity(tree)
    assert c['time'] == 'O(n)'


def test_infer_quadratic():
    tree = ast.parse(
        "def process(d):\n"
        "    for x in d:\n"
        "        for y in d:\n"
        "            print(x, y)"
    )
    c = LLMAlgorithmSynthesizer._infer_complexity(tree)
    assert c['time'] == 'O(n^2)'
