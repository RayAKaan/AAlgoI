from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aalgoi.types import AlgorithmSpec, ProblemSpec, SolveResult


class Algorithm(ABC):
    spec: AlgorithmSpec

    @abstractmethod
    def run(self, spec: ProblemSpec) -> Any:
        ...

    def execute(self, spec: ProblemSpec) -> SolveResult:
        import time
        t0 = time.time()
        try:
            output = self.run(spec)
            elapsed = (time.time() - t0) * 1000
            ok = True
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return SolveResult(
                output=None,
                ok=False,
                algorithm=self.spec.name,
                error=str(e),
                time_ms=elapsed,
            )
        from aalgoi.types import ValidationResult
        validated = False
        validation = ValidationResult()
        if self.spec.validator is not None:
            try:
                valid = self.spec.validator(spec.inputs, output)
                validation.passed = valid
                validation.oracle_match = valid
                validated = True
            except Exception as e:
                validation.errors.append(str(e))
        return SolveResult(
            output=output,
            ok=ok,
            algorithm=self.spec.name,
            validated=validated,
            validation=validation,
            complexity=self.spec.complexity,
            time_ms=elapsed,
        )
