from __future__ import annotations

from typing import Any

from aalgoi.rl.env import Environment
from aalgoi.rl.policy import Policy


class Trainer:
    def __init__(self, n_actions: int = 10) -> None:
        self.env = Environment()
        self.policy = Policy(n_actions)
        self.episodes = 0

    def train_step(self, state: list[float], action: int, reward: float) -> None:
        self.policy.update(state, action, reward)
        self.episodes += 1

    def save(self, path: str) -> None:
        import json
        data = {"weights": self.policy._weights.tolist(), "episodes": self.episodes}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        self.policy._weights = np.array(data["weights"], dtype=np.float32)
        self.episodes = data["episodes"]
