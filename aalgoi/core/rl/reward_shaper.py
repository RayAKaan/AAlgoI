import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RewardShaper:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.weights = {
            "quality": self.config.get("quality_weight", 0.5),
            "speed": self.config.get("speed_weight", 0.3),
            "memory": self.config.get("memory_weight", 0.1),
            "novelty": self.config.get("novelty_weight", 0.1),
        }
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        self.reward_scale = self.config.get("reward_scale", 10.0)
        self.failure_penalty = -10.0
        self.success_bonus = 2.0
        self.improvement_bonus = 5.0
        self.novel_bonus = 3.0
        self.discovery_bonus = 50.0

    def compute(self, success: bool, elapsed: float, data_size: int,
                algo_name: str = "", metrics: dict = None) -> float:
        if not success:
            return self.failure_penalty

        reward = self.reward_scale * 0.5

        if data_size > 0 and elapsed > 0:
            expected_log = max(1.0, data_size * 0.03)
            ratio = expected_log / (elapsed * 1000 + 0.001)
            speed_bonus = min(10.0, ratio * 3.0)
        else:
            speed_bonus = 0.0
        reward += speed_bonus

        size_bonus = min(2.0, 0.5 * (data_size / 10000))
        reward += size_bonus

        if elapsed < 0.001 and data_size > 100:
            reward += 3.0

        if metrics and isinstance(metrics, dict):
            if "accuracy" in metrics:
                acc = float(metrics["accuracy"])
                acc_reward = (acc - 0.5) * 20.0
                reward += acc_reward
            elif "r2_score" in metrics:
                r2 = float(metrics["r2_score"])
                r2_reward = r2 * 10.0
                reward += r2_reward
            elif "f1" in metrics:
                f1 = float(metrics["f1"])
                f1_reward = (f1 - 0.5) * 15.0
                reward += f1_reward

        return reward

    def compute_reward(self, is_valid: bool, metrics: dict[str, Any],
                       context: dict[str, Any], history: dict[str, Any] = None,
                       discovered_algorithm: dict | None = None) -> float:
        if not is_valid:
            return self.failure_penalty

        reward = 0.0
        reward += self.weights["quality"] * self._compute_quality_reward(metrics, context)
        reward += self.weights["speed"] * self._compute_speed_reward(metrics, context)
        reward += self.weights["memory"] * self._compute_memory_reward(metrics, context)
        reward += self.weights["novelty"] * self._compute_novelty_reward(metrics, history)
        reward *= self.reward_scale
        reward += self.success_bonus

        if history and self._is_improvement(metrics, history):
            reward += self.improvement_bonus

        if discovered_algorithm:
            reward += self.discovery_bonus
            novelty_score = discovered_algorithm.get("novelty_score", 0)
            reward += novelty_score * 10.0

        return reward

    def _compute_quality_reward(self, metrics: dict, context: dict) -> float:
        quality = metrics.get("quality_score", 0.0)
        accuracy = metrics.get("accuracy", 1.0)
        return float(np.clip(quality * accuracy * 10, 0, 10))

    def _compute_speed_reward(self, metrics: dict, context: dict) -> float:
        actual = metrics.get("wall_time_ms", 0) / 1000.0
        budget = context.get("time_budget_ms", 1000) / 1000.0
        if actual == 0:
            return 10.0
        ratio = actual / budget
        if ratio <= 0.5:
            return 10.0
        elif ratio <= 1.0:
            return 10.0 * (1.0 - ratio)
        else:
            return -5.0 * (ratio - 1.0)

    def _compute_memory_reward(self, metrics: dict, context: dict) -> float:
        memory = metrics.get("memory_mb", 0)
        budget = context.get("memory_budget_mb", 1024)
        if memory == 0:
            return 5.0
        ratio = memory / budget
        if ratio <= 0.5:
            return 5.0
        elif ratio <= 1.0:
            return 5.0 * (1.0 - ratio)
        else:
            return -3.0 * (ratio - 1.0)

    def _compute_novelty_reward(self, metrics: dict, history: dict) -> float:
        if not history:
            return 0.0
        algorithms = metrics.get("algorithms", [])
        past = history.get("algorithm_combinations", [])
        combo = tuple(sorted(a.name if hasattr(a, "name") else str(a) for a in algorithms))
        if combo not in past:
            return self.novel_bonus
        count = past.count(combo)
        return self.novel_bonus * np.exp(-count / 10.0)

    def _is_improvement(self, metrics: dict, history: dict) -> bool:
        current = metrics.get("quality_score", 0)
        best = history.get("best_score", 0)
        return current > best * 1.05


class AdaptiveRewardShaper(RewardShaper):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.base_weights = dict(self.weights)

    def compute_reward(self, is_valid, metrics, context, history=None):
        priority = context.get("priority", "balanced")
        if priority == "speed":
            self.weights.update({"quality": 0.2, "speed": 0.6, "memory": 0.1, "novelty": 0.1})
        elif priority == "accuracy":
            self.weights.update({"quality": 0.7, "speed": 0.1, "memory": 0.1, "novelty": 0.1})
        elif priority == "memory":
            self.weights.update({"quality": 0.3, "speed": 0.2, "memory": 0.4, "novelty": 0.1})
        else:
            self.weights.update(self.base_weights)
        return super().compute_reward(is_valid, metrics, context, history)


class CurriculumRewardShaper(RewardShaper):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.training_progress = 0.0

    def set_progress(self, progress: float):
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def compute_reward(self, is_valid, metrics, context, history=None):
        base = super().compute_reward(is_valid, metrics, context, history)
        if self.training_progress < 0.3 and base > 0:
            base *= 1.5
        elif self.training_progress > 0.7:
            quality = metrics.get("quality_score", 0)
            if quality < 0.9:
                base *= 0.5
        return base
