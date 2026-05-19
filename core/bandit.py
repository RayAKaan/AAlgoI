
import math
import random
from typing import Dict, List, Any, Optional


class UCB1Bandit:
    def __init__(self, algorithm_names: List[str], epsilon: float = 0.2,
                 epsilon_decay: float = 0.99, epsilon_min: float = 0.05):
        self.algorithm_names = algorithm_names
        self.counts: Dict[str, int] = {name: 0 for name in algorithm_names}
        self.rewards: Dict[str, float] = {name: 0.0 for name in algorithm_names}
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.total_trials = 0

    def select(self, candidates: Optional[List[str]] = None) -> str:
        pool = candidates if candidates else self.algorithm_names
        if not pool:
            return self.algorithm_names[0]

        if random.random() < self.epsilon:
            return random.choice(pool)

        total = max(self.total_trials, 1)
        def ucb1(name: str) -> float:
            n = max(self.counts.get(name, 0), 1)
            avg_reward = self.rewards.get(name, 0.0) / n
            exploration = math.sqrt(2 * math.log(total) / n)
            return avg_reward + exploration

        return max(pool, key=ucb1)

    def update(self, name: str, reward: float):
        if name not in self.counts:
            return
        self.counts[name] += 1
        self.rewards[name] += reward
        self.total_trials += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_exploration(self):
        self.epsilon = min(0.2, self.epsilon + 0.1)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "total_trials": self.total_trials,
            "algorithm_stats": {
                name: {
                    "count": self.counts[name],
                    "avg_reward": self.rewards[name] / self.counts[name] if self.counts[name] > 0 else 0.0
                }
                for name in self.algorithm_names
            }
        }
