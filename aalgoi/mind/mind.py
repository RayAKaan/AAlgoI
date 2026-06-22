from __future__ import annotations

from pathlib import Path
from typing import Any

from aalgoi.algorithms.registry import get_registry
from aalgoi.eval.benchmark_suite import BenchmarkSuite, core_v1
from aalgoi.eval.reports import format_report
from aalgoi.kg.graph import KnowledgeGraph
from aalgoi.kg.seed import seed_from_registry
from aalgoi.kg.store import Store
from aalgoi.mind.learning import Trainer
from aalgoi.selection.planner import Planner
from aalgoi.types import BenchmarkReport, SolveResult


class Mind:
    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = str(Path.home() / ".aalgoi" / "mind")
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)

        self.registry = get_registry()
        self.kg = KnowledgeGraph()
        seed_from_registry(self.kg, self.registry)
        self.store = Store(self.path / "store.db")
        self.planner = Planner(self.registry, self.kg, self.store)
        self._solve_count = 0
        self._success_count = 0

    @property
    def algorithms(self) -> list[str]:
        return self.registry.get_names()

    @property
    def principles(self) -> list[str]:
        seen: set[str] = set()
        for name in self.registry.get_names():
            spec = self.registry.get_spec(name)
            if spec:
                seen.update(spec.principles)
        return sorted(seen)

    @property
    def problems(self) -> list[str]:
        return sorted(set(spec.task.value for name in self.registry.get_names() if (spec := self.registry.get_spec(name))))

    def solve(self, problem_text: str, data: Any = None, mode: str = "deterministic") -> SolveResult:
        result = self.planner.solve(problem_text, data, mode)
        self._solve_count += 1
        if result.ok:
            self._success_count += 1
        return result

    def train(self, mode: str = "bandit", suite: str = "core-v1", **kwargs: Any) -> dict:
        try:
            trainer = Trainer(self.planner)
            return trainer.train(mode=mode, suite=suite, **kwargs)
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

    def benchmark(self, suite: str = "core-v1") -> BenchmarkReport:
        bs = core_v1() if suite == "core-v1" else BenchmarkSuite(suite)
        return bs.run()

    def checkpoint(self, name: str | None = None) -> str | None:
        from aalgoi.mind.checkpoint import CheckpointManager
        cm = CheckpointManager(self.path / "checkpoints")
        return cm.save(name)

    def rollback(self, target: str = "last_good") -> dict:
        from aalgoi.mind.checkpoint import CheckpointManager
        cm = CheckpointManager(self.path / "checkpoints")
        return cm.restore(target)

    def share(self) -> int:
        outbox = self.path / "outbox"
        outbox.mkdir(exist_ok=True)
        return len(list(outbox.glob("*.json")))

    def receive(self) -> dict:
        inbox = self.path / "inbox"
        inbox.mkdir(exist_ok=True)
        return {"updates_processed": len(list(inbox.glob("*.json"))), "algorithms_imported": 0}

    def status(self) -> str:
        algos = self.algorithms
        rate = self._success_count / max(self._solve_count, 1) * 100
        lines = [
            f"Algorithms:  {len(algos)}",
            f"Principles:  {len(self.principles)}",
            f"Problems:    {len(self.problems)}",
            f"Solved:      {self._solve_count}",
            f"Success rate: {rate:.0f}%",
        ]
        width = max(len(l) for l in lines) + 4
        top = "+" + "-" * width + "+"
        bot = "+" + "-" * width + "+"
        body = "\n".join(f"| {l:<{width}} |" for l in ["[ Algorithmic Mind ]"] + lines)
        return f"{top}\n{body}\n{bot}"
