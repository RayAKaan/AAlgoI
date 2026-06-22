from __future__ import annotations

import time
from typing import Any

from aalgoi.algorithms.registry import AlgorithmRegistry, get_registry
from aalgoi.kg.graph import KnowledgeGraph
from aalgoi.kg.queries import QueryEngine
from aalgoi.kg.seed import seed_from_registry
from aalgoi.kg.store import Store
from aalgoi.kg.updater import Updater
from aalgoi.problems.oracles import evaluate as oracle_evaluate
from aalgoi.problems.parser import ProblemParser
from aalgoi.selection.rankers import RuleRanker
from aalgoi.types import (
    CandidateScore, DecisionEvent, ProblemSpec, ProblemTask, SolveMode,
    SolveResult, ValidationResult,
)


class Planner:
    def __init__(
        self,
        registry: AlgorithmRegistry | None = None,
        kg: KnowledgeGraph | None = None,
        store: Store | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.kg = kg or KnowledgeGraph()
        self.store = store or Store()
        if kg is None:
            seed_from_registry(self.kg, self.registry)
        self.queries = QueryEngine(self.kg, self.store)
        self.updater = Updater(self.kg, self.store)
        self.parser = ProblemParser()
        self.ranker = RuleRanker(self.registry, self.queries)

    def solve(
        self,
        problem_text: str,
        data: Any = None,
        mode: str = "deterministic",
    ) -> SolveResult:
        trace: list[DecisionEvent] = []
        t0 = time.time()

        spec = self._parse(problem_text, data, trace)
        candidates = self._retrieve(spec, trace)
        ranked = self._rank(spec, candidates, trace)
        result = self._execute(spec, ranked, trace)

        elapsed = (time.time() - t0) * 1000
        result.time_ms = elapsed
        result.trace = trace
        result.mode = mode

        self.updater.record_run(
            spec, result.algorithm or "none",
            result.ok, result.validated, elapsed, result.error,
        )
        return result

    def _parse(self, problem_text: str, data: Any, trace: list[DecisionEvent]) -> ProblemSpec:
        t0 = time.time()
        spec = self.parser.parse(problem_text, data)
        trace.append(DecisionEvent("parse", f"task={spec.task.value} confidence={spec.confidence:.2f}", (time.time() - t0) * 1000))
        return spec

    def _retrieve(self, spec: ProblemSpec, trace: list[DecisionEvent]) -> list[str]:
        t0 = time.time()
        candidates = self.queries.candidates_for(spec)
        trace.append(DecisionEvent("retrieve", f"found {len(candidates)} candidates for {spec.task.value}", (time.time() - t0) * 1000))
        return candidates

    def _rank(self, spec: ProblemSpec, candidates: list[str], trace: list[DecisionEvent]) -> list[CandidateScore]:
        t0 = time.time()
        scored = self.ranker.rank(spec, candidates)
        trace.append(DecisionEvent("rank", f"top: {scored[0].algorithm if scored else 'none'}", (time.time() - t0) * 1000))
        return scored

    def _execute(self, spec: ProblemSpec, ranked: list[CandidateScore], trace: list[DecisionEvent]) -> SolveResult:
        t0 = time.time()
        best_error: str | None = None
        for cs in ranked:
            try:
                algo = self.registry.create(cs.algorithm)
            except KeyError:
                continue
            t_algo = time.time()
            result = algo.execute(spec)
            t_algo_elapsed = (time.time() - t_algo) * 1000
            trace.append(DecisionEvent("execute", f"{cs.algorithm}: ok={result.ok} validated={result.validated}", t_algo_elapsed))
            if result.ok:
                validation = ValidationResult()
                try:
                    validated = oracle_evaluate(spec.task, spec.inputs, result.output)
                    validation.passed = validated
                    validation.oracle_match = validated
                except Exception as e:
                    validation.errors.append(str(e))
                    validated = False
                return SolveResult(
                    output=result.output,
                    ok=validated,
                    algorithm=result.algorithm,
                    validated=True,
                    validation=validation,
                    candidates=ranked,
                    complexity=result.complexity,
                    time_ms=(time.time() - t0) * 1000,
                    confidence=cs.score,
                )
            if result.error:
                best_error = result.error
        return SolveResult(
            output=None,
            ok=False,
            error=best_error or "No algorithm produced a valid result",
            candidates=ranked,
            time_ms=(time.time() - t0) * 1000,
        )


solve = Planner().solve
