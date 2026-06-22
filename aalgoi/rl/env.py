from __future__ import annotations

from typing import Any

from aalgoi.types import ProblemSpec


class Environment:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._done = False
        self._reward = 0.0

    def reset(self) -> dict:
        self._state = {}
        self._done = False
        self._reward = 0.0
        return self._state

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        self._state["last_action"] = action
        self._done = True
        return self._state, self._reward, self._done, {}

    def close(self) -> None:
        pass
