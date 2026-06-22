from __future__ import annotations

from typing import Any

from aalgoi.kg.graph import KnowledgeGraph
from aalgoi.kg.store import Store
from aalgoi.types import ProblemSpec, ProblemTask


class QueryEngine:
    def __init__(self, kg: KnowledgeGraph, store: Store) -> None:
        self.kg = kg
        self.store = store

    def candidates_for(self, spec: ProblemSpec) -> list[str]:
        candidates = self.kg.find_candidates(spec.task)
        scored = []
        for name in candidates:
            perf = self.store.get_performance(name, spec.task.value)
            score = 0.5
            if perf["run_count"] > 0:
                score = (perf["successes"] / perf["run_count"]) * 0.7 + 0.3
            scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored]

    def explain_algorithm(self, name: str) -> str | None:
        return self.kg.explain(name)

    def find_similar_tasks(self, spec: ProblemSpec) -> list[ProblemTask]:
        return self.kg.find_similar_tasks(spec)

    def get_performance_profile(self, algorithm: str, task: str) -> dict:
        return self.store.get_performance(algorithm, task)
