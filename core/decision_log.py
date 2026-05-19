
import json
import time
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Decision:
    context: Dict[str, Any]
    candidates: List[str]
    chosen: str
    confidence: float
    reason: str
    timestamp: float = 0.0
    outcome_success: Optional[bool] = None
    wall_time_ms: Optional[float] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DecisionLog:
    def __init__(self, path: str = "audit.jsonl"):
        self.path = path
        self.recent_decisions: List[Decision] = []
        self.max_in_memory = 1000

    def record(self, decision: Decision):
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > self.max_in_memory:
            self.recent_decisions = self.recent_decisions[-self.max_in_memory:]

        try:
            with open(self.path, 'a') as f:
                line = json.dumps(asdict(decision), default=str)
                f.write(line + '\n')
        except IOError:
            pass

    def get_recent(self, n: int = 10) -> List[Decision]:
        return self.recent_decisions[-n:]

    def get_last(self) -> Optional[Decision]:
        if self.recent_decisions:
            return self.recent_decisions[-1]
        return None

    def get_by_algorithm(self, algo_name: str, n: int = 10) -> List[Decision]:
        matches = [d for d in self.recent_decisions if d.chosen == algo_name]
        return matches[-n:]

    def get_stats(self) -> Dict[str, Any]:
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
