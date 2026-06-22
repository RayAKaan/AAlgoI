from __future__ import annotations

import functools
from typing import Any

from aalgoi.types import AlgorithmSpec, ProblemSpec, ProblemTask, SolveResult


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._algorithms: dict[str, AlgorithmSpec] = {}
        self._by_task: dict[ProblemTask, list[str]] = {}
        self._implementations: dict[str, type] = {}

    def register(self, algo_cls: type, spec: AlgorithmSpec) -> None:
        name = spec.name
        if name in self._algorithms:
            raise KeyError(f"Algorithm '{name}' is already registered")
        self._algorithms[name] = spec
        self._by_task.setdefault(spec.task, []).append(name)
        self._implementations[name] = algo_cls
        algo_cls.spec = spec

    def get_spec(self, name: str) -> AlgorithmSpec | None:
        return self._algorithms.get(name)

    def get_names(self) -> list[str]:
        return list(self._algorithms.keys())

    def candidates_for(self, task: ProblemTask) -> list[str]:
        return list(self._by_task.get(task, []))

    def candidates_for_spec(self, spec: ProblemSpec) -> list[str]:
        return self.candidates_for(spec.task)

    def create(self, name: str):
        cls = self._implementations.get(name)
        if cls is None:
            raise KeyError(f"Algorithm '{name}' not found in registry")
        return cls()

    def __contains__(self, name: str) -> bool:
        return name in self._algorithms

    def __len__(self) -> int:
        return len(self._algorithms)

    def __repr__(self) -> str:
        return f"AlgorithmRegistry({len(self)} algorithms, {len(self._by_task)} tasks)"


_REGISTRY: AlgorithmRegistry | None = None


def get_registry() -> AlgorithmRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = AlgorithmRegistry()
    return _REGISTRY


def algorithm(spec: AlgorithmSpec):
    def decorator(cls):
        reg = get_registry()
        reg.register(cls, spec)
        return cls
    return decorator
