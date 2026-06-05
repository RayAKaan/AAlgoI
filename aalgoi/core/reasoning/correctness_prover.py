from dataclasses import dataclass
from typing import Any


@dataclass
class CorrectnessProof:
    is_correct: bool
    proof_type: str
    proof_text: str
    confidence: float
    is_formal: bool
    counterexample: str | None = None
    stress_iterations: int = 0
    edge_cases_passed: int = 0
    edge_cases_total: int = 0


class CorrectnessProver:
    ORACLE_MAP = {
        "integers": lambda expected, actual: sorted(expected) == sorted(actual),
        "graph": lambda expected, actual: expected == actual,
        "text": lambda expected, actual: expected == actual,
        "feature_matrix": lambda expected, actual: expected == actual,
    }

    def prove(
        self,
        candidate_code: str,
        problem_text: str,
        data: Any,
        examples: list[dict] | None = None,
        essence: "ProblemEssence | None" = None,
    ) -> CorrectnessProof:
        if not examples:
            return CorrectnessProof(
                is_correct=False,
                proof_type="none",
                proof_text="No examples provided for verification",
                confidence=0.0,
                is_formal=False,
            )

        passed = 0
        failures = []
        for ex in examples:
            try:
                actual = self._execute_safely(candidate_code, ex.get("input", data))
                expected = ex.get("expected_output", ex.get("output"))
                if actual == expected:
                    passed += 1
                else:
                    failures.append(
                        f"Input: {ex.get('input')}, Expected: {expected}, Got: {actual}"
                    )
            except Exception as e:
                failures.append(f"Input: {ex.get('input')}, Error: {e}")

        confidence = passed / len(examples) if examples else 0.0

        return CorrectnessProof(
            is_correct=passed == len(examples),
            proof_type="empirical",
            proof_text=(
                f"Passed {passed}/{len(examples)} provided examples.\n"
                + (f"Failures: {failures[:3]}" if failures else "All examples passed.")
            ),
            confidence=confidence,
            is_formal=False,
            counterexample=failures[0] if failures else None,
            stress_iterations=0,
            edge_cases_passed=passed,
            edge_cases_total=len(examples),
        )

    def _execute_safely(self, code: str, input_data: Any) -> Any:
        try:
            from aalgoi.core.sandbox import execute_sandboxed
            return execute_sandboxed(code, input_data)
        except ImportError:
            restricted_globals = {
                "__builtins__": {
                    "list": list, "dict": dict, "set": set, "tuple": tuple,
                    "len": len, "range": range, "enumerate": enumerate,
                    "sorted": sorted, "min": min, "max": max, "sum": sum,
                    "abs": abs, "int": int, "float": float, "str": str,
                    "bool": bool, "print": print, "isinstance": isinstance,
                    "True": True, "False": False, "None": None,
                }
            }
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            for val in local_vars.values():
                if callable(val) and not isinstance(val, type):
                    return val(input_data)
            return None
