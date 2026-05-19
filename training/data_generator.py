import random
import numpy as np
import logging
from typing import Any, Dict, Tuple

from core.problem_spec import ProblemSpec, ProblemType

logger = logging.getLogger(__name__)


class CurriculumLevel:
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class SyntheticDataGenerator:
    """Curriculum-aware data generator for progressive RL training."""

    def __init__(self):
        self.level = CurriculumLevel.BEGINNER
        self.stats = {"problems_generated": 0}

    def set_level(self, level: int):
        self.level = level
        logger.info(f"Curriculum advanced to level {level}")

    def generate_sorting(self) -> Tuple[ProblemSpec, list]:
        if self.level == CurriculumLevel.BEGINNER:
            size = random.randint(10, 100)
            scenario = random.choice(["random", "sorted"])
        elif self.level == CurriculumLevel.INTERMEDIATE:
            size = random.randint(100, 1000)
            scenario = random.choice(["random", "nearly_sorted", "duplicates"])
        elif self.level == CurriculumLevel.ADVANCED:
            size = random.randint(1000, 5000)
            scenario = random.choice(["reverse", "nearly_sorted", "random"])
        else:
            size = random.randint(5000, 20000)
            scenario = random.choice(["reverse", "nearly_sorted", "duplicates", "random"])

        if scenario == "random":
            data = np.random.randint(0, 100000, size).tolist()
        elif scenario == "sorted":
            data = sorted(np.random.randint(0, 100000, size).tolist())
        elif scenario == "nearly_sorted":
            data = sorted(np.random.randint(0, 100000, size).tolist())
            num_swaps = max(1, size // random.randint(15, 20))
            for _ in range(num_swaps):
                i, j = random.sample(range(size), 2)
                data[i], data[j] = data[j], data[i]
        elif scenario == "reverse":
            data = sorted(np.random.randint(0, 100000, size).tolist(), reverse=True)
        else:
            data = np.random.choice([1, 10, 100, 1000], size).tolist()

        spec = ProblemSpec(
            name=f"sort_{scenario}_lvl{self.level}",
            problem_type=ProblemType.SORTING,
        )
        self.stats["problems_generated"] += 1
        return spec, data

    def generate_pathfinding(self) -> Tuple[ProblemSpec, Dict]:
        if self.level == CurriculumLevel.BEGINNER:
            nodes = random.randint(5, 20)
            density = 0.3
        elif self.level == CurriculumLevel.INTERMEDIATE:
            nodes = random.randint(20, 50)
            density = 0.4
        elif self.level == CurriculumLevel.ADVANCED:
            nodes = random.randint(50, 100)
            density = 0.5
        else:
            nodes = random.randint(100, 200)
            density = 0.6

        graph = {str(i): {} for i in range(nodes)}
        for i in range(nodes):
            for j in range(i + 1, nodes):
                if random.random() < density:
                    weight = (
                        1
                        if self.level <= CurriculumLevel.INTERMEDIATE
                        else random.randint(1, 50)
                    )
                    graph[str(i)][str(j)] = weight

        if all(len(v) == 0 for v in graph.values()):
            graph["0"]["1"] = 1

        spec = ProblemSpec(
            name=f"path_lvl{self.level}",
            problem_type=ProblemType.PATHFINDING,
        )
        return spec, {"graph": graph, "start": "0", "end": str(nodes - 1)}

    def generate_optimization(self) -> Tuple[ProblemSpec, Dict]:
        if self.level == CurriculumLevel.BEGINNER:
            items_count = random.randint(10, 30)
            capacity = random.randint(50, 150)
        elif self.level == CurriculumLevel.INTERMEDIATE:
            items_count = random.randint(30, 80)
            capacity = random.randint(150, 400)
        else:
            items_count = random.randint(80, 200)
            capacity = random.randint(400, 1000)

        items = [
            {"value": random.randint(10, 200), "weight": random.randint(1, 100)}
            for _ in range(items_count)
        ]

        spec = ProblemSpec(
            name=f"knapsack_lvl{self.level}",
            problem_type=ProblemType.OPTIMIZATION,
        )
        return spec, {"items": items, "capacity": capacity}
