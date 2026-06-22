from dataclasses import dataclass


@dataclass
class Explanation:
    algorithm_name: str
    summary: str
    complexity: str
    steps: list[str]
    best_for: str
    source: str
    detail_level: str
    execution_time_ms: float = 0.0
