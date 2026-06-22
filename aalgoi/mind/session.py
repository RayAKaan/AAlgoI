from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from aalgoi.mind.mind import Mind
from aalgoi.types import SolveResult


@contextmanager
def session(path: str | None = None) -> Iterator["MindSession"]:
    mind = Mind(path)
    ms = MindSession(mind)
    try:
        yield ms
    finally:
        pass


class MindSession:
    def __init__(self, mind: Mind) -> None:
        self.mind = mind

    def solve(self, problem: str, data: Any = None, mode: str = "deterministic") -> SolveResult:
        return self.mind.solve(problem, data, mode)

    def status(self) -> str:
        return self.mind.status()
