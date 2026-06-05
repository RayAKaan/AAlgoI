"""
Curriculum Learning for Progressive Training
Starts easy, increases difficulty as agent improves.
"""

import logging
from typing import Any

from aalgoi.core.problem_spec import ProblemSpec
from training.data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Manages training difficulty progression.
    Level 1.0 (easy sorting) to 10.0 (research-level optimization).
    """

    def __init__(self):
        self.difficulty_level = 1.0
        self.success_history: list = []

    def update_difficulty(self, success_rate: float):
        """Adjust difficulty based on recent success rate."""
        self.success_history.append(success_rate)

        if len(self.success_history) >= 10:
            recent_avg = sum(self.success_history[-10:]) / 10

            if recent_avg > 0.9 and self.difficulty_level < 10.0:
                self.difficulty_level += 0.5
                logger.info("Curriculum: Advanced to level %.1f", self.difficulty_level)
                self.success_history.clear()

            elif recent_avg < 0.5 and self.difficulty_level > 1.0:
                self.difficulty_level -= 0.5
                logger.info("Curriculum: Reduced to level %.1f", self.difficulty_level)
                self.success_history.clear()

    def generate_problem(self) -> tuple[ProblemSpec, Any]:
        """Generate problem appropriate for current difficulty level."""
        gen = SyntheticDataGenerator()

        if self.difficulty_level <= 3:
            spec, data = gen.generate_sorting()
        elif self.difficulty_level <= 7:
            spec, data = gen.generate_pathfinding()
        else:
            spec, data = gen.generate_optimization()

        return spec, data
