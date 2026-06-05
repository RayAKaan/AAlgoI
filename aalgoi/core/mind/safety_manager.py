import json
import logging
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aalgoi.core.mind.rl_mind import AlgorithmicMind

logger = logging.getLogger(__name__)


class MindSafetyManager:
    CHECKPOINT_INTERVAL = 50
    DEGRADATION_WARNING_SIGMA = 2.0
    DEGRADATION_ROLLBACK_SIGMA = 3.0
    MIN_HISTORY_FOR_DETECTION = 30

    def __init__(self, user_dir: Path) -> None:
        self.user_dir = Path(user_dir)
        self._solve_count = 0
        self._history: list[dict] = []

    @property
    def solve_count(self) -> int:
        return self._solve_count

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def auto_checkpoint(self, mind: "AlgorithmicMind", count: int) -> None:
        self._solve_count = count
        solve_time = getattr(mind, "last_solve_time", None)
        if solve_time is None:
            solve_time = time.time()
        self._history.append({"solve_count": count, "time": solve_time})

        if count % self.CHECKPOINT_INTERVAL != 0:
            return

        recent = self._history[-self.MIN_HISTORY_FOR_DETECTION:] if len(self._history) >= self.MIN_HISTORY_FOR_DETECTION else self._history
        times = [h["time"] for h in recent]
        rolling_mean = statistics.mean(times) if times else 0.0
        rolling_std = statistics.stdev(times) if len(times) >= 2 else 0.0

        checkpoint = {
            "solve_count": count,
            "timestamp": time.time(),
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
        }

        ckpt_dir = self.user_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_{count}.json"
        try:
            with open(ckpt_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
        except OSError as e:
            logger.warning("Checkpoint save failed at count %d: %s", count, e)

    def rollback(self, target: str = "last_good") -> None:
        if target != "last_good":
            raise ValueError(f"Unknown rollback target: {target}")

        ckpt_dir = self.user_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            raise ValueError("No checkpoint directory found")

        checkpoints = sorted(
            ckpt_dir.glob("checkpoint_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )
        if not checkpoints:
            raise ValueError("No checkpoints available for rollback")

        best = checkpoints[0]
        try:
            with open(best) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load checkpoint {best}: {e}")

        logger.info(
            "Rolled back to checkpoint %s (solve count %d)",
            best.name, data.get("solve_count", "?"),
        )

    def detect_quality_degradation(self) -> bool:
        if len(self._history) < self.MIN_HISTORY_FOR_DETECTION:
            return False

        recent = self._history[-self.MIN_HISTORY_FOR_DETECTION:]
        latest = recent[-1]["time"]
        times = [h["time"] for h in recent]
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) >= 2 else 0.0

        if std == 0.0:
            return False

        if latest > mean + self.DEGRADATION_ROLLBACK_SIGMA * std:
            logger.warning(
                "Auto-rollback triggered: latest time %.4f "
                "exceeds mean %.4f + %.1f*std %.4f",
                latest, mean, self.DEGRADATION_ROLLBACK_SIGMA, std,
            )
            try:
                self.rollback("last_good")
            except ValueError as e:
                logger.warning("Auto-rollback failed: %s", e)
            return True

        if latest > mean + self.DEGRADATION_WARNING_SIGMA * std:
            logger.warning(
                "Degradation warning: latest time %.4f "
                "exceeds mean %.4f + %.1f*std %.4f",
                latest, mean, self.DEGRADATION_WARNING_SIGMA, std,
            )
            return True

        return False
