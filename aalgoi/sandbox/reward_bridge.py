"""
aalgoi.sandbox.reward_bridge — Reward computation bridging existing infrastructure.

Priority:
  1. core.rl.reward_shaper.RewardShaper — uses the production reward formula
  2. AdaptiveRewardShaper if available
  3. Manual fallback: success + speed + difficulty + algo_match bonus
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional


class RewardBridge:
    """
    Computes shaped rewards using the existing reward infrastructure.
    Gracefully falls back if the production shaper isn't available.
    """

    def __init__(self):
        self._shaper = None
        for cls_path in [
            "core.rl.reward_shaper.AdaptiveRewardShaper",
            "core.rl.reward_shaper.RewardShaper",
        ]:
            try:
                module_path, cls_name = cls_path.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[cls_name])
                self._shaper = getattr(mod, cls_name)()
                break
            except Exception:
                continue

    def compute(
        self,
        success: bool,
        time_ms: float,
        difficulty: int,
        chosen_algo: str,
        expected_algo: Optional[str],
        confidence: float,
    ) -> float:
        if self._shaper is not None:
            try:
                return self._compute_via_shaper(success, time_ms, chosen_algo)
            except Exception:
                pass
        return self._compute_manual(success, time_ms, difficulty, chosen_algo, expected_algo, confidence)

    def _compute_via_shaper(self, success: bool, time_ms: float, chosen_algo: str) -> float:
        speed_score   = 1.0 / (1.0 + time_ms / 1000.0)
        return self._shaper.compute(
            success=success,
            elapsed=time_ms / 1000.0,
            data_size=0,
            algo_name=chosen_algo,
        )

    def _compute_manual(
        self,
        success: bool,
        time_ms: float,
        difficulty: int,
        chosen_algo: str,
        expected_algo: Optional[str],
        confidence: float,
    ) -> float:
        reward = 1.0 if success else -0.15

        if success:
            speed = max(0.0, 1.0 - time_ms / 5000.0)
            reward += 0.15 * speed

        if success:
            reward += difficulty * 0.02

        if expected_algo and chosen_algo == expected_algo:
            reward += 0.25

        if not success and confidence > 0.5:
            reward -= 0.1 * confidence

        if success and confidence > 0.6:
            reward += 0.05

        return float(np.clip(reward, -0.5, 2.0))
