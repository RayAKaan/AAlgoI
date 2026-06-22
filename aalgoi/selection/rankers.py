from __future__ import annotations

from aalgoi.algorithms.registry import AlgorithmRegistry
from aalgoi.kg.queries import QueryEngine
from aalgoi.types import CandidateScore, ProblemSpec


class RuleRanker:
    def __init__(self, registry: AlgorithmRegistry, queries: QueryEngine | None = None) -> None:
        self.registry = registry
        self.queries = queries

    def rank(self, spec: ProblemSpec, candidates: list[str]) -> list[CandidateScore]:
        scored = []
        for name in candidates:
            spec_algo = self.registry.get_spec(name)
            if spec_algo is None:
                continue
            score = 0.5
            if spec_algo.exact:
                score += 0.2
            if spec_algo.deterministic:
                score += 0.1
            if spec_algo.task == spec.task:
                score += 0.2
            if self.queries:
                perf = self.queries.get_performance_profile(name, spec.task.value)
                if perf["run_count"] > 0:
                    score = score * 0.5 + 0.5 * (perf["successes"] / perf["run_count"])
            scored.append(CandidateScore(
                algorithm=name,
                task=spec_algo.task,
                score=score,
                source="rule",
                reason=f"exact={spec_algo.exact} deterministic={spec_algo.deterministic}",
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored


class KGRanker:
    def __init__(self, queries: QueryEngine) -> None:
        self.queries = queries

    def rank(self, spec: ProblemSpec, candidates: list[str]) -> list[CandidateScore]:
        return []


class SupervisedRanker:
    def rank(self, spec: ProblemSpec, candidates: list[str]) -> list[CandidateScore]:
        return []
