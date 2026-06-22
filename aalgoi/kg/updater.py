from __future__ import annotations

from typing import Any

from aalgoi.kg.graph import KnowledgeGraph
from aalgoi.kg.store import Store
from aalgoi.types import ProblemSpec


class Updater:
    def __init__(self, kg: KnowledgeGraph, store: Store) -> None:
        self.kg = kg
        self.store = store

    def record_run(self, spec: ProblemSpec, algorithm: str, success: bool, validated: bool, time_ms: float, error: str | None = None) -> None:
        self.store.record_run(spec.id, spec.task.value, algorithm, success, validated, time_ms, error)

    def record_failure(self, spec: ProblemSpec, algorithm: str, reason: str) -> None:
        self.store.record_failure(spec.id, algorithm, reason)

    def update_performance(self, algorithm: str, task: str, metrics: dict) -> None:
        pass
