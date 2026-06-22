from __future__ import annotations

import math
import random


class UCB1Bandit:
    def __init__(self, epsilon: float = 0.2, epsilon_decay: float = 0.99, epsilon_min: float = 0.05) -> None:
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.counts: dict[str, int] = {}
        self.rewards: dict[str, float] = {}
        self.total_trials = 0

    def select(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("No candidates to select from")
        for c in candidates:
            if c not in self.counts:
                self.counts[c] = 0
                self.rewards[c] = 0.0
        if random.random() < self.epsilon:
            return random.choice(candidates)
        total = sum(self.counts.values())
        scores = {}
        for c in candidates:
            if self.counts[c] == 0:
                return c
            avg = self.rewards[c] / self.counts[c]
            bonus = math.sqrt(2 * math.log(total) / self.counts[c])
            scores[c] = avg + bonus
        return max(scores, key=scores.get)

    def update(self, name: str, reward: float) -> None:
        if name in self.counts:
            self.counts[name] += 1
            self.rewards[name] += reward
        else:
            self.counts[name] = 1
            self.rewards[name] = reward
        self.total_trials += 1

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get_stats(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "total_trials": self.total_trials,
            "algorithms": {
                name: {
                    "count": self.counts.get(name, 0),
                    "avg_reward": self.rewards.get(name, 0.0) / max(self.counts.get(name, 1), 1),
                }
                for name in set(list(self.counts.keys()) + list(self.rewards.keys()))
            },
        }
