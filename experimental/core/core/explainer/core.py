import json
import logging
import time
from typing import Any

from aalgoi.core.explainer.model import Explanation
from aalgoi.core.explainer.templates import EXPLANATION_TEMPLATES
from aalgoi.core.problem_spec import ProblemSpec

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, llm_client: Any | None = None,
                 default_detail: str = "short"):
        self.llm_client = llm_client
        self.default_detail = default_detail

    def explain(self, algorithm_name: str,
                detail: str | None = None,
                problem_spec: ProblemSpec | None = None,
                context: dict | None = None) -> Explanation:
        start = time.perf_counter()
        detail = detail or self.default_detail

        if detail == "short":
            result = self._template_explain(algorithm_name, context)
        elif detail == "detailed" and self.llm_client:
            result = self._llm_explain(algorithm_name, problem_spec, context)
        else:
            result = self._template_explain(algorithm_name, context)

        result.execution_time_ms = (time.perf_counter() - start) * 1000
        return result

    def explain_pipeline(self, pipeline_names: list[str],
                         detail: str | None = None,
                         problem_spec: ProblemSpec | None = None) -> list[Explanation]:
        return [
            self.explain(name, detail=detail, problem_spec=problem_spec)
            for name in pipeline_names
        ]

    def _template_explain(self, algorithm_name: str,
                          context: dict | None = None) -> Explanation:
        normalized = algorithm_name.lower().replace("_primitive", "").replace("_", "_")
        template = EXPLANATION_TEMPLATES.get(normalized)

        if template is None:
            template = EXPLANATION_TEMPLATES.get(
                next((k for k in EXPLANATION_TEMPLATES if k in normalized), None),
                {
                    "summary": f"{algorithm_name} is a custom algorithm that processes data according to its specific logic.",
                    "complexity": "Unknown (not profiled)",
                    "steps": [
                        f"Execute {algorithm_name} on the input data",
                        "Return the processed result"
                    ],
                    "best_for": "Custom use cases"
                }
            )

        explanation = Explanation(
            algorithm_name=algorithm_name,
            summary=template["summary"],
            complexity=template["complexity"],
            steps=list(template["steps"]),
            best_for=template["best_for"],
            source="template",
            detail_level="short"
        )

        if context:
            explanation.summary += f" (Context: {context.get('task', 'general')})"

        return explanation

    def _llm_explain(self, algorithm_name: str,
                     problem_spec: ProblemSpec | None = None,
                     context: dict | None = None) -> Explanation:
        if not self.llm_client:
            return self._template_explain(algorithm_name, context)

        try:
            prompt = self._build_explain_prompt(algorithm_name, problem_spec, context)
            response = self.llm_client.generate(prompt)
            content = response.get("explanation", "") if isinstance(response, dict) else str(response)

            return Explanation(
                algorithm_name=algorithm_name,
                summary=content,
                complexity="See explanation",
                steps=[],
                best_for="",
                source="llm",
                detail_level="detailed"
            )
        except Exception as e:
            logger.warning(f"LLM explanation failed for {algorithm_name}: {e}")
            return self._template_explain(algorithm_name, context)

    def _build_explain_prompt(self, algorithm_name: str,
                              problem_spec: ProblemSpec | None = None,
                              context: dict | None = None) -> str:
        parts = [f"Explain the {algorithm_name} algorithm in detail."]
        if problem_spec:
            parts.append(f"Problem: {problem_spec.to_dict()}")
        if context:
            parts.append(f"Context: {json.dumps(context)}")
        parts.append("Include: how it works, time/space complexity, and when to use it.")
        return "\n".join(parts)

    def list_available_explanations(self) -> list[str]:
        return sorted(EXPLANATION_TEMPLATES.keys())
