"""Transparent SolveResult proxy — behaves like the raw output value."""

from __future__ import annotations

from typing import Any


class SolveResult:
    """
    Transparent result proxy.

    Use it like the raw output value:
        result = solve("sort", [3, 1, 2])
        result[0]    # → 1
        len(result)  # → 3
        result == [1, 2, 3]  # → True

    Or access metadata:
        result.algorithm   # → "tim_sort"
        result.complexity  # → "O(n log n)"
        result.explain()
    """

    def __init__(
        self,
        output: Any = None,
        code: str | None = None,
        algorithm: str | None = None,
        complexity: str | None = None,
        principle: str | None = None,
        time_ms: float = 0.0,
        is_novel: bool = False,
        confidence: float = 0.0,
        iterations: int = 0,
        error: str | None = None,
    ) -> None:
        self.output = output
        self.code = code
        self.algorithm = algorithm
        self.complexity = complexity
        self.principle = principle
        self.time_ms = time_ms
        self.is_novel = is_novel
        self.confidence = confidence
        self.iterations = iterations
        self.error = error

    # ── Succes / failure ────────────────────────────────────────────

    @property
    def ok(self) -> bool:
        """True if the solve produced a valid output."""
        return self.error is None and self.output is not None

    @property
    def success(self) -> bool:
        """Backward-compat alias for ``.ok``."""
        return self.ok

    def __bool__(self) -> bool:
        return self.ok

    # ── Transparent passthrough to .output ──────────────────────────

    def __str__(self) -> str:
        if self.output is not None:
            return str(self.output)
        if self.error:
            return f"Error: {self.error}"
        return ""

    def __repr__(self) -> str:
        from aalgoi._status import box

        if self.error:
            return box([f"❌ {self.error}"], title="SolveResult")

        if self.output is None:
            return box(["⚠  No output"], title="SolveResult")

        status = "🆕" if self.is_novel else ""
        algo = self.algorithm or "?"
        conf = f" ({self.confidence:.0%})" if self.confidence else ""
        lines = [
            f"Algorithm: {algo}{status}{conf}",
            f"Complexity: {self.complexity or '?'}",
            f"Principle:  {self.principle or '?'}",
            f"Time:       {self.time_ms:.1f}ms",
            f"Iterations: {self.iterations}",
        ]
        return box(lines, title="SolveResult")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SolveResult):
            return self.output == other.output
        return self.output == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        try:
            return hash(self.output)
        except TypeError:
            return hash(str(self.output))

    def __iter__(self):
        if self.output is None:
            return iter([])
        return iter(self.output)

    def __getitem__(self, index):
        if self.output is None:
            raise IndexError("No output")
        try:
            return self.output[index]
        except TypeError:
            raise IndexError("Output not indexable")

    def __len__(self) -> int:
        if self.output is None:
            return 0
        try:
            return len(self.output)
        except TypeError:
            return 1

    def __contains__(self, item) -> bool:
        if self.output is None:
            return False
        try:
            return item in self.output
        except TypeError:
            return self.output == item

    def __add__(self, other):
        if self.output is None:
            return NotImplemented
        try:
            return self.output + other
        except TypeError:
            return NotImplemented

    def __radd__(self, other):
        if self.output is None:
            return NotImplemented
        try:
            return other + self.output
        except TypeError:
            return NotImplemented

    def __int__(self) -> int:
        if self.output is None:
            return 0
        try:
            return int(self.output)
        except (TypeError, ValueError):
            return 0

    def __float__(self) -> float:
        if self.output is None:
            return 0.0
        try:
            return float(self.output)
        except (TypeError, ValueError):
            return 0.0

    # ── Explanation ──────────────────────────────────────────────────

    def explain(self) -> str:
        """Return a human-readable explanation of the solution."""
        from aalgoi._status import box

        lines = []

        if self.error:
            lines.append(f"Error: {self.error}")
            return "\n".join(lines)

        name = self.algorithm or "Unknown"
        complexity = self.complexity or "unknown"
        principle = self.principle or "unknown"
        confidence = self.confidence or 0.0

        lines.append(f"Algorithm: {name}")
        lines.append(f"Complexity: {complexity}")
        lines.append(f"Principle: {principle}")
        lines.append(f"Confidence: {confidence:.0%}")

        if self.is_novel:
            lines.append("Status: 🆕 Novel algorithm discovered!")

        if self.time_ms:
            lines.append(f"Time: {self.time_ms:.1f}ms")

        if self.iterations:
            lines.append(f"Cognitive steps: {self.iterations}")

        if self.code:
            lines.append("")
            lines.append("Generated code:")
            lines.append(self.code)

        return "\n".join(lines)
