
import json
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Decision:
    context: dict[str, Any]
    candidates: list[str]
    chosen: str
    confidence: float
    reason: str
    timestamp: float = 0.0
    outcome_success: bool | None = None
    wall_time_ms: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DecisionLog:
    def __init__(self, path: str = "audit.jsonl"):
        self.path = path
        self.recent_decisions: list[Decision] = []
        self.max_in_memory = 1000

    def record(self, decision: Decision) -> None:
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > self.max_in_memory:
            self.recent_decisions = self.recent_decisions[-self.max_in_memory:]

        try:
            with open(self.path, 'a') as f:
                line = json.dumps(asdict(decision), default=str)
                f.write(line + '\n')
        except OSError:
            pass

    def get_recent(self, n: int = 10) -> list[Decision]:
        return self.recent_decisions[-n:]

    def get_last(self) -> Decision | None:
        if self.recent_decisions:
            return self.recent_decisions[-1]
        return None

    def get_by_algorithm(self, algo_name: str, n: int = 10) -> list[Decision]:
        matches = [d for d in self.recent_decisions if d.chosen == algo_name]
        return matches[-n:]

    def get_stats(self) -> dict[str, Any]:
        if not self.recent_decisions:
            return {"total_decisions": 0}

        from collections import Counter
        algo_counts = Counter(d.chosen for d in self.recent_decisions)
        success_count = sum(1 for d in self.recent_decisions if d.outcome_success is True)
        failure_count = sum(1 for d in self.recent_decisions if d.outcome_success is False)

        return {
            "total_decisions": len(self.recent_decisions),
            "by_algorithm": dict(algo_counts),
            "successful": success_count,
            "failed": failure_count,
            "avg_confidence": sum(d.confidence for d in self.recent_decisions) / len(self.recent_decisions),
            "last_decision_time": self.recent_decisions[-1].timestamp if self.recent_decisions else None
        }
