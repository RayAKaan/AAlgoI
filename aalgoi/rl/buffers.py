from __future__ import annotations

from typing import Any


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self._buffer: list[dict[str, Any]] = []

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        self._buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def sample(self, batch_size: int = 32) -> list[dict[str, Any]]:
        import random
        if len(self._buffer) < batch_size:
            return self._buffer.copy()
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
