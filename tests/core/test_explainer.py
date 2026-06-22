from unittest.mock import MagicMock

import pytest
from aalgoi.core.explainer import Explainer, Explanation, EXPLANATION_TEMPLATES
from aalgoi.core.problem_spec import ProblemSpec, ProblemType


class TestTemplateExplain:
    def test_known_algorithm(self):
        e = Explainer()
        result = e._template_explain("quicksort")
        assert isinstance(result, Explanation)
        assert result.algorithm_name == "quicksort"
        assert result.summary == EXPLANATION_TEMPLATES["quicksort"]["summary"]
        assert result.complexity == EXPLANATION_TEMPLATES["quicksort"]["complexity"]
        assert result.best_for == EXPLANATION_TEMPLATES["quicksort"]["best_for"]
        assert result.source == "template"
        assert result.detail_level == "short"

    def test_unknown_algorithm_uses_fallback(self):
        e = Explainer()
        result = e._template_explain("nonexistent_algo_42")
        assert result.algorithm_name == "nonexistent_algo_42"
        assert "custom algorithm" in result.summary.lower()
        assert result.source == "template"

    def test_partial_match_in_templates(self):
        e = Explainer()
        result = e._template_explain("heap_sort_primitive")
        assert "heapsort" in result.summary.lower() or "max-heap" in result.summary.lower()
        assert result.source == "template"

    def test_binary_search_partial(self):
        e = Explainer()
        result = e._template_explain("binary_search_v2")
        assert result.algorithm_name == "binary_search_v2"
        assert "binary search" in result.summary.lower()

    def test_with_context(self):
        e = Explainer()
        result = e._template_explain("quicksort", context={"task": "sorting"})
        assert "sorting" in result.summary
        assert result.source == "template"

    def test_with_empty_context(self):
        e = Explainer()
        result = e._template_explain("quicksort", context={})
        assert result.source == "template"

    def test_normalize_removes_primitive_suffix(self):
        e = Explainer()
        result = e._template_explain("knn_primitive")
        assert "knn" in result.algorithm_name or "K-Nearest" in result.summary

    @pytest.mark.parametrize("algo", [
        "mergesort", "heap_sort", "bfs", "dfs", "kmeans", "dbscan",
        "logistic_regression", "svm", "pca", "greedy", "lcs"
    ])
    def test_various_known_algos(self, algo):
        e = Explainer()
        result = e._template_explain(algo)
        assert result.summary
        assert len(result.steps) > 0
        assert result.best_for
        assert result.source == "template"

    def test_result_is_independent(self):
        e = Explainer()
        r1 = e._template_explain("quicksort")
        r2 = e._template_explain("quicksort")
        assert r1.summary == r2.summary
        assert r1.steps is not r2.steps


class TestExplainMethod:
    def test_explain_default_detail(self):
        e = Explainer()
        result = e.explain("quicksort")
        assert isinstance(result, Explanation)
        assert result.algorithm_name == "quicksort"
        assert result.source == "template"

    def test_explain_detailed_without_llm_falls_back(self):
        e = Explainer()
        result = e.explain("quicksort", detail="detailed")
        assert result.source == "template"

    def test_explain_short_explicit(self):
        e = Explainer()
        result = e.explain("quicksort", detail="short")
        assert result.source == "template"

    def test_explain_detailed_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {"explanation": "LLM generated explanation"}
        e = Explainer(llm_client=mock_llm)
        result = e.explain("quicksort", detail="detailed")
        assert result.source == "llm"
        assert result.summary == "LLM generated explanation"
        mock_llm.generate.assert_called_once()

    def test_explain_with_context(self):
        e = Explainer()
        result = e.explain("quicksort", context={"task": "sorting"})
        assert "sorting" in result.summary

    def test_execution_time_is_set(self):
        e = Explainer()
        result = e.explain("quicksort")
        assert result.execution_time_ms >= 0.0


class TestExplainPipeline:
    def test_pipeline_multiple_names(self):
        e = Explainer()
        results = e.explain_pipeline(["quicksort", "mergesort", "bfs"])
        assert len(results) == 3
        assert all(isinstance(r, Explanation) for r in results)
        assert [r.algorithm_name for r in results] == ["quicksort", "mergesort", "bfs"]

    def test_pipeline_empty(self):
        e = Explainer()
        results = e.explain_pipeline([])
        assert results == []

    def test_pipeline_with_detail(self):
        e = Explainer()
        results = e.explain_pipeline(["quicksort"], detail="short")
        assert len(results) == 1
        assert results[0].source == "template"


class TestBuildExplainPrompt:
    def test_basic_prompt(self):
        e = Explainer()
        prompt = e._build_explain_prompt("quicksort")
        assert "Explain the quicksort algorithm" in prompt
        assert "time/space complexity" in prompt

    def test_with_problem_spec(self):
        e = Explainer()
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        prompt = e._build_explain_prompt("quicksort", problem_spec=spec)
        assert "Explain the quicksort algorithm" in prompt
        assert "time/space complexity" in prompt

    def test_with_context(self):
        e = Explainer()
        prompt = e._build_explain_prompt("quicksort", context={"task": "sort"})
        assert "task" in prompt and "sort" in prompt

    def test_with_all_params(self):
        e = Explainer()
        spec = ProblemSpec(name="t", problem_type=ProblemType.SORTING)
        prompt = e._build_explain_prompt("quicksort", problem_spec=spec, context={"key": "val"})
        assert "Explain" in prompt
        assert "time/space" in prompt


class TestLLMExplain:
    def test_without_llm_falls_back(self):
        e = Explainer()
        result = e._llm_explain("quicksort")
        assert result.source == "template"

    def test_with_llm_returns_llm_explanation(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {"explanation": "Narrative explanation"}
        e = Explainer(llm_client=mock_llm)
        result = e._llm_explain("quicksort")
        assert result.source == "llm"
        assert result.summary == "Narrative explanation"
        assert result.algorithm_name == "quicksort"

    def test_with_llm_string_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "String response"
        e = Explainer(llm_client=mock_llm)
        result = e._llm_explain("quicksort")
        assert result.source == "llm"
        assert result.summary == "String response"

    def test_llm_error_falls_back(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API down")
        e = Explainer(llm_client=mock_llm)
        result = e._llm_explain("quicksort")
        assert result.source == "template"


class TestListAvailable:
    def test_returns_sorted_keys(self):
        e = Explainer()
        result = e.list_available_explanations()
        assert result == sorted(result)
        assert len(result) > 30
        assert "quicksort" in result
        assert "mergesort" in result
        assert "knn" in result
        assert "kmeans" in result

    def test_immutable(self):
        e = Explainer()
        result = e.list_available_explanations()
        original_count = len(result)
        assert original_count == len(EXPLANATION_TEMPLATES)


class TestExplanationDataclass:
    def test_defaults(self):
        exp = Explanation(
            algorithm_name="test",
            summary="desc",
            complexity="O(1)",
            steps=["do it"],
            best_for="everything",
            source="template",
            detail_level="short"
        )
        assert exp.execution_time_ms == 0.0

    def test_fields(self):
        exp = Explanation(
            algorithm_name="algo",
            summary="sum",
            complexity="O(n)",
            steps=["a", "b"],
            best_for="all",
            source="llm",
            detail_level="detailed",
            execution_time_ms=1.5
        )
        assert exp.algorithm_name == "algo"
        assert exp.execution_time_ms == 1.5
        assert len(exp.steps) == 2
