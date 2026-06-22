from __future__ import annotations

from typing import Any

from aalgoi.eval.benchmark_suite import core_v1
from aalgoi.selection.bandit import UCB1Bandit
from aalgoi.selection.planner import Planner


class Trainer:
    def __init__(self, planner: Planner) -> None:
        self.planner = planner
        self.bandit = UCB1Bandit()

    def train(self, mode: str = "bandit", suite: str = "core-v1", **kwargs: Any) -> dict:
        bs = core_v1() if suite == "core-v1" else None
        if bs is None:
            return {"status": "error", "error": f"Unknown suite: {suite}"}
        if mode == "bandit":
            return self._train_bandit(bs)
        if mode == "supervised":
            return self._train_supervised(bs)
        return {"status": "unavailable", "error": f"Unknown training mode: {mode}"}

    def _train_bandit(self, bs: Any) -> dict:
        problems = getattr(bs, "_problems", None)
        if not problems:
            return {"status": "unavailable", "error": "no problems in suite"}
        for spec in problems:
            candidates = self.planner.queries.candidates_for(spec)
            if candidates:
                chosen = self.bandit.select(candidates)
                result = self.planner.solve(spec.task.value, spec.inputs)
                reward = 1.0 if result.ok else 0.0
                self.bandit.update(chosen, reward)
        self.bandit.decay_epsilon()
        stats = self.bandit.get_stats()
        return {"status": "ok", "mode": "bandit", "total_trials": stats["total_trials"], "epsilon": stats["epsilon"]}

    def _train_supervised(self, bs: Any) -> dict:
        return {"status": "unavailable", "error": "supervised training not yet implemented"}
