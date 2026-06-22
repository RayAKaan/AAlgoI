from __future__ import annotations

from typing import Any

from aalgoi.errors import AAlgoIError
from aalgoi.mind.mind import Mind
from aalgoi.mind.session import session
from aalgoi.selection.planner import Planner
from aalgoi.types import SolveResult, SolveMode


def solve(
    problem_text: str,
    data: Any = None,
    mode: str = "deterministic",
    mind: Mind | None = None,
) -> SolveResult:
    if mind is not None:
        return mind.solve(problem_text, data, mode=mode)
    planner = Planner()
    return planner.solve(problem_text, data, mode=mode)


__all__ = ["solve", "session", "Mind"]
