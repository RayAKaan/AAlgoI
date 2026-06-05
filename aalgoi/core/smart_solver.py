"""
User-friendly wrapper that accepts plain English questions.
"""

import logging
from typing import Any, Dict, Optional

from aalgoi.core.question_parser import QuestionParser
from aalgoi.core.problem_spec import ProblemType

logger = logging.getLogger(__name__)


class SmartSolver:
    """
    Natural language interface to aalgoi.

    Usage:
        solver = SmartSolver()
        result = solver.ask("Sort these numbers fastest possible", [5, 3, 8, 1])
        result = solver.ask("Train Word2Vec on medical corpus with 200 dimensions", corpus)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.parser = QuestionParser(use_transformer=False)
        self._config = config or {}
        self._solver = None

    @property
    def solver(self):
        if self._solver is None:
            from aalgoi.pipeline import UniversalSolver
            self._solver = UniversalSolver(config=self._config)
        return self._solver

    def ask_with_spec(self, spec, data: Any = None, **kwargs) -> Dict:
        """
        Same as ask() but accepts a pre-built ProblemSpec.
        Skips NL parsing. Used by aalgoi.transformer.solve().
        """
        if kwargs:
            for key, value in kwargs.items():
                if key == "priority" and value:
                    spec.constraints.append(f"priority={value}")
                elif key not in spec.inputs:
                    spec.inputs[key] = value

        result = self.solver.solve(spec, data)

        if result.get("success"):
            question = spec.description or "problem"
            result["answer"] = self._format_answer(question, result)

        return result

    def ask(self, question: str, data: Any = None, **kwargs) -> Dict:
        """
        Main interface: question + data -> answer

        Args:
            question: Natural language question
            data: Input data (list, dict, array, etc.)
            **kwargs: Override parameters (priority, etc.)

        Returns:
            Result dict with answer, explanation, algorithm used
        """
        spec = self.parser.parse(question, data)

        if spec.problem_type == ProblemType.UNKNOWN:
            return {
                "success": False,
                "result": None,
                "error": (
                    "I couldn't determine what kind of problem this is. "
                    "Please include a more specific keyword like "
                    "'sort', 'path', 'train', 'classify', 'cluster', "
                    "'blur', or 'optimize'."
                ),
                "algorithm": None,
                "time_ms": 0,
            }

        if kwargs:
            for key, value in kwargs.items():
                if key == "priority" and value:
                    spec.constraints.append(f"priority={value}")
                elif key not in spec.inputs:
                    spec.inputs[key] = value

        result = self.solver.solve(spec, data)

        if result.get("success"):
            result["answer"] = self._format_answer(question, result)

        return result

    def _format_answer(self, question: str, result: Dict) -> str:
        algo = result.get("algorithm", "unknown")
        time_ms = result.get("time_ms", 0)

        answer = f"Solved using {algo} in {time_ms:.2f}ms."

        output = result.get("result")
        if isinstance(output, dict):
            if "vocabulary_size" in output:
                answer += f" Trained model with {output['vocabulary_size']} words."
            elif "selected" in output:
                answer += f" Selected {len(output['selected'])} items, value={output.get('value', 'N/A')}."

        return answer
