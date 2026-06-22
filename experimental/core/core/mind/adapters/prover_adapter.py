from typing import Any

from aalgoi.core.reasoning.correctness_prover import CorrectnessProof, CorrectnessProver


class ProverAdapter:
    def __init__(self, prover: CorrectnessProver | None = None) -> None:
        self._prover = prover or CorrectnessProver()

    def prove(
        self,
        candidate_code: str,
        problem_text: str,
        data: Any,
        examples: list[dict] | None = None,
        essence: Any = None,
    ) -> CorrectnessProof:
        return self._prover.prove(
            candidate_code=candidate_code,
            problem_text=problem_text,
            data=data,
            examples=examples,
            essence=essence,
        )

    def stress_test(
        self,
        candidate_code: str,
        brute_force_code: str | None,
        essence: Any,
        iterations: int = 1000,
    ) -> CorrectnessProof:
        return self._prover.prove(
            candidate_code=candidate_code,
            problem_text="",
            data=None,
            examples=[],
            essence=essence,
        )
