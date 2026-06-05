"""MindSession — context manager for an ephemeral solve session."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aalgoi._core import Mind
from aalgoi._result import SolveResult
from aalgoi._status import box


class MindSession:
    """
    Ephemeral solve session — created via ``with aalgoi.session():``.

    Records solve history and prints a summary on exit.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mind = Mind(path)
        self._history: list[dict] = []
        self._start = time.time()

    def solve(
        self,
        problem: str,
        data: Any = None,
        *,
        examples: list[dict] | None = None,
        max_iterations: int = 50,
        time_limit: float = 30.0,
        hint: str | None = None,
    ) -> SolveResult:
        result = self._mind.solve(
            problem, data,
            examples=examples,
            max_iterations=max_iterations,
            time_limit=time_limit,
            hint=hint,
        )
        self._history.append({
            "problem": problem,
            "algorithm": result.algorithm,
            "confidence": result.confidence,
            "time_ms": result.time_ms,
            "ok": result.ok,
        })
        return result

    def learn(
        self,
        problem: str,
        data: Any = None,
        *,
        expected: Any = None,
        examples: list[dict] | None = None,
        max_iterations: int = 50,
        time_limit: float = 30.0,
    ) -> SolveResult:
        result = self.solve(
            problem, data,
            examples=examples,
            max_iterations=max_iterations,
            time_limit=time_limit,
        )
        if expected is not None and result.ok:
            if result.output != expected:
                pass  # will track correctness in future iterations
        return result

    def status(self) -> str:
        elapsed = time.time() - self._start
        lines = [
            f"Duration:    {elapsed:.1f}s",
            f"Solved:      {len(self._history)} problems",
        ]
        if self._history:
            ok_count = sum(1 for h in self._history if h["ok"])
            lines.append(f"Successful:  {ok_count}")
            avg_time = sum(h["time_ms"] for h in self._history) / len(self._history)
            lines.append(f"Avg time:    {avg_time:.1f}ms")
        return box(lines, title="Mind Session")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._summarize()

    def _summarize(self) -> None:
        elapsed = time.time() - self._start
        n = len(self._history)
        ok_n = sum(1 for h in self._history if h["ok"])
        print(
            f"Session done: {ok_n}/{n} solved in {elapsed:.1f}s"
        )
