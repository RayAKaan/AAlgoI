from __future__ import annotations


class RewardShaper:
    def __init__(self, success_reward: float = 1.0, failure_penalty: float = -0.1) -> None:
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty

    def compute(self, ok: bool, time_ms: float = 0.0) -> float:
        if ok:
            bonus = max(0.0, 1.0 - time_ms / 10000.0)
            return self.success_reward + bonus
        return self.failure_penalty
