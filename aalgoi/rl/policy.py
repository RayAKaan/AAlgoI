from __future__ import annotations

import numpy as np


class Policy:
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self._weights = np.zeros(n_actions, dtype=np.float32)

    def select_action(self, state: list[float]) -> int:
        return int(np.argmax(self._weights))

    def update(self, state: list[float], action: int, reward: float) -> None:
        self._weights[action] += reward * 0.01
