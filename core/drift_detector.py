
from collections import deque
import numpy as np
from typing import List, Optional


class DriftDetector:
    def __init__(self, window: int = 100, threshold: float = 0.05):
        self.window_size = window
        self.threshold = threshold
        self.scores: deque = deque(maxlen=window)
        self.reference_mean: Optional[float] = None
        self.drift_count = 0
        self.last_drift_time = 0.0
        self._index = 0

    def update(self, score: float, timestamp: float = 0.0) -> bool:
        self.scores.append(score)
        self._index += 1

        if len(self.scores) < self.window_size:
            return False

        current_mean = float(np.mean(self.scores))

        if self.reference_mean is None:
            self.reference_mean = current_mean
            return False

        deviation = abs(current_mean - self.reference_mean)

        if deviation > self.threshold:
            self.drift_count += 1
            self.last_drift_time = timestamp
            self.reference_mean = current_mean
            return True

        if self._index % self.window_size == 0:
            self.reference_mean = 0.9 * self.reference_mean + 0.1 * current_mean

        return False

    def get_drift_rate(self) -> float:
        if self._index == 0:
            return 0.0
        return self.drift_count / (self._index // self.window_size + 1)

    def reset(self):
        self.scores.clear()
        self.reference_mean = None
        self.drift_count = 0
        self.last_drift_time = 0.0
        self._index = 0

    def get_stats(self) -> dict:
        return {
            "window_size": self.window_size,
            "current_size": len(self.scores),
            "current_mean": float(np.mean(self.scores)) if self.scores else None,
            "reference_mean": self.reference_mean,
            "drift_count": self.drift_count,
            "drift_rate": self.get_drift_rate()
        }
