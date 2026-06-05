from dataclasses import dataclass, field
from typing import Any
import time
import hashlib
import torch

from aalgoi.core.mind.rl_mind import AlgorithmicMind
from aalgoi.core.mind.cognitive_actions import (
    CognitiveAction, ActionParams, ActionResult, ActionHandler,
)
from aalgoi.core.mind.mind_state import MindState, build_data_profile
from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph
from aalgoi.core.mind.safety_manager import MindSafetyManager


@dataclass
class ThinkingSession:
    problem_text: str
    data: Any
    problem_signature: str
    examples: list[dict] = field(default_factory=list)
    max_iterations: int = 50
    time_limit_seconds: float = 30.0
    actions_taken: list[CognitiveAction] = field(default_factory=list)
    action_results: list[ActionResult] = field(default_factory=list)
    solutions_tried: list[dict] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)
    current_best_code: str | None = None
    current_best_algo: str | None = None
    current_best_verified: bool = False
    current_best_is_novel: bool = False
    current_best_correctness: float = 0.0
    start_time: float = field(default_factory=time.time)
    phase: str = "understanding"
    consecutive_failures: int = 0
    last_improvement_step: int = 0


@dataclass
class UniversalSolution:
    code: str | None
    output: Any
    principle_applied: str
    time_complexity: str
    space_complexity: str
    is_optimal: bool
    is_novel: bool
    correctness_proof: "CorrectnessProof | None"
    why_this_works: str
    complexity_proof: str
    mathematical_essence: str
    actions_taken: list[CognitiveAction]
    solve_time_ms: float
    iterations: int


class MindSolvingLoop:
    def __init__(
        self,
        mind: AlgorithmicMind,
        kg: AlgorithmicKnowledgeGraph,
        action_handler: ActionHandler,
        safety: MindSafetyManager,
    ) -> None:
        self.mind = mind
        self.kg = kg
        self.handler = action_handler
        self.safety = safety

    def solve(
        self,
        problem_text: str,
        data: Any,
        examples: list[dict] | None = None,
        max_iterations: int = 50,
    ) -> UniversalSolution:
        session = self._init_session(problem_text, data, examples)
        session.max_iterations = max_iterations
        state = self._init_state(session)

        while self._should_continue(session):
            available = self._get_available_actions(state, session)

            action, params, log_prob = self.mind.select_action(
                state, available, self.kg,
            )

            result = self.handler.dispatch(action, params, state, session)

            state = state.update_after_action(action, result)
            session.actions_taken.append(action)
            session.action_results.append(result)

            self._update_best(session, result)

            reward = self._compute_reward(action, result, state, session)
            self.mind.store_transition(
                state, action, reward, state, False, log_prob,
            )

            if action == CognitiveAction.ACCEPT_SOLUTION:
                if session.current_best_verified:
                    break

            if action == CognitiveAction.BACKTRACK:
                state = self._backtrack(state, session)

            if self._should_auto_accept(session):
                break

        solve_time_ms = (time.time() - session.start_time) * 1000

        if session.current_best_is_novel and session.current_best_verified:
            self.handler.dispatch(
                CognitiveAction.RECORD_DISCOVERY,
                ActionParams(action=CognitiveAction.RECORD_DISCOVERY),
                state,
                session,
            )

        for failed in session.failed_approaches:
            self.kg.record_failure(
                failed, session.problem_signature, "failed_in_solving_loop"
            )

        self._online_train()
        self.safety.auto_checkpoint(self.mind, self.safety.solve_count)

        return self._build_solution(session, state, solve_time_ms)

    def _init_session(
        self,
        problem_text: str,
        data: Any,
        examples: list[dict] | None,
    ) -> ThinkingSession:
        sig = hashlib.sha256(
            problem_text.encode()
        ).hexdigest()[:16]
        return ThinkingSession(
            problem_text=problem_text,
            data=data,
            problem_signature=sig,
            examples=examples or [],
        )

    def _init_state(self, session: ThinkingSession) -> MindState:
        return MindState(
            problem_text=session.problem_text,
            problem_signature=session.problem_signature,
            data_features=build_data_profile(session.data),
        )

    def _should_continue(self, session: ThinkingSession) -> bool:
        elapsed = time.time() - session.start_time
        return (
            elapsed < session.time_limit_seconds
            and len(session.actions_taken) < session.max_iterations
        )

    def _get_available_actions(
        self,
        state: MindState,
        session: ThinkingSession,
    ) -> list[CognitiveAction]:
        step = len(session.actions_taken)
        available = []

        available.append(CognitiveAction.BACKTRACK)
        available.append(CognitiveAction.REQUEST_SIMPLIFY)

        available.extend([
            CognitiveAction.DECOMPOSE_PROBLEM,
            CognitiveAction.IDENTIFY_STRUCTURE,
            CognitiveAction.EXTRACT_CONSTRAINTS,
            CognitiveAction.FIND_INVARIANT,
            CognitiveAction.ESTIMATE_COMPLEXITY,
        ])

        if step >= 2 or state.identified_structure is not None:
            available.extend([
                CognitiveAction.QUERY_SIMILAR,
                CognitiveAction.QUERY_PRINCIPLE,
                CognitiveAction.QUERY_ALGORITHMS,
                CognitiveAction.QUERY_FAILURES,
                CognitiveAction.QUERY_COMPLEXITY,
            ])
            session.phase = "retrieval"

        if step >= 4 or session.current_best_code is not None:
            available.extend([
                CognitiveAction.SELECT_ALGORITHM,
                CognitiveAction.MODIFY_ALGORITHM,
                CognitiveAction.COMBINE_ALGORITHMS,
                CognitiveAction.APPLY_OPTIMIZATION,
                CognitiveAction.SYNTHESIZE_NEW,
                CognitiveAction.DECOMPOSE_RECURSIVE,
            ])
            session.phase = "construction"

        if session.current_best_code:
            available.extend([
                CognitiveAction.TEST_EXAMPLES,
                CognitiveAction.STRESS_TEST,
                CognitiveAction.CHECK_EDGE_CASES,
                CognitiveAction.VERIFY_COMPLEXITY,
                CognitiveAction.PROVE_CORRECTNESS,
            ])
            session.phase = "verification"

        if session.current_best_code and session.current_best_correctness > 0.8:
            available.append(CognitiveAction.ACCEPT_SOLUTION)

        if session.current_best_is_novel and session.current_best_verified:
            available.append(CognitiveAction.RECORD_DISCOVERY)

        return available

    def _update_best(
        self,
        session: ThinkingSession,
        result: ActionResult,
    ) -> None:
        if result.solution_code and result.correctness > session.current_best_correctness:
            session.current_best_code = result.solution_code
            session.current_best_algo = (
                result.output.get("algorithm") if isinstance(result.output, dict) else None
            )
            session.current_best_correctness = result.correctness
            session.current_best_is_novel = result.is_novel_algorithm
            session.last_improvement_step = len(session.actions_taken)
            session.consecutive_failures = 0

            session.current_best_verified = (
                result.correctness > 0.95
                and any(
                    a in [CognitiveAction.STRESS_TEST, CognitiveAction.PROVE_CORRECTNESS]
                    for a in session.actions_taken[-5:]
                )
            )
        elif not result.success:
            session.consecutive_failures += 1

        if result.action in [
            CognitiveAction.STRESS_TEST,
            CognitiveAction.PROVE_CORRECTNESS,
        ] and result.success:
            session.current_best_verified = True

    def _should_auto_accept(self, session: ThinkingSession) -> bool:
        return (
            session.current_best_verified
            and session.current_best_correctness > 0.95
            and len(session.actions_taken) >= 8
        )

    def _should_backtrack(
        self,
        state: MindState,
        session: ThinkingSession,
    ) -> bool:
        steps_since_improvement = len(session.actions_taken) - session.last_improvement_step
        if steps_since_improvement >= 8:
            return True

        if len(session.actions_taken) >= 5:
            recent = session.actions_taken[-5:]
            for action in set(recent):
                if recent.count(action) >= 3:
                    return True

        return False

    def _backtrack(
        self,
        state: MindState,
        session: ThinkingSession,
    ) -> MindState:
        if session.solutions_tried:
            last = session.solutions_tried.pop()
            algo_name = last.get("algorithm")
            if algo_name and algo_name not in session.failed_approaches:
                session.failed_approaches.append(algo_name)
        session.current_best_code = None
        session.current_best_algo = None
        session.current_best_verified = False
        return state

    def _compute_reward(
        self,
        action: CognitiveAction,
        result: ActionResult,
        state: MindState,
        session: ThinkingSession,
    ) -> float:
        reward = result.reward_signal

        if result.correctness > state.correctness_confidence:
            reward += 0.3

        if result.time_complexity and state.current_complexity != "unknown":
            if self._is_better_complexity(result.time_complexity, state.current_complexity):
                reward += 0.5

        if result.is_novel_algorithm and result.correctness > 0.8:
            reward += 2.0

        if action in [CognitiveAction.STRESS_TEST, CognitiveAction.PROVE_CORRECTNESS]:
            if result.success:
                reward += 0.5
            else:
                reward -= 0.3

        recent = session.actions_taken[-5:] if len(session.actions_taken) >= 5 else []
        if recent.count(action) >= 3:
            reward -= 0.5

        if session.consecutive_failures >= 3:
            reward -= 0.2

        reward -= 0.01

        return reward

    @staticmethod
    def _is_better_complexity(new: str, old: str) -> bool:
        order = [
            "O(1)", "O(log n)", "O(n)", "O(n log n)",
            "O(n^2)", "O(2^n)", "unknown",
        ]
        try:
            return order.index(new) < order.index(old)
        except ValueError:
            return False

    def _online_train(self) -> None:
        optimizer = torch.optim.AdamW(
            self.mind.parameters(), lr=3e-4
        )
        self.mind.train_on_trajectory(optimizer)

    def _build_solution(
        self,
        session: ThinkingSession,
        state: MindState,
        solve_time_ms: float,
    ) -> UniversalSolution:
        from aalgoi.core.reasoning.correctness_prover import CorrectnessProof

        proof = CorrectnessProof(
            is_correct=session.current_best_verified,
            proof_type="empirical",
            proof_text=(
                f"Verified after {len(session.actions_taken)} cognitive actions"
            ),
            confidence=session.current_best_correctness,
            is_formal=False,
        )

        if session.current_best_algo and session.current_best_verified:
            self.kg.record_success(
                session.current_best_algo,
                session.problem_signature,
                session.current_best_correctness,
            )
        elif session.current_best_algo:
            self.kg.record_failure(
                session.current_best_algo,
                session.problem_signature,
                "verification_failed",
            )

        return UniversalSolution(
            code=session.current_best_code,
            output=None,
            principle_applied=state.identified_principle or "unknown",
            time_complexity=state.current_complexity,
            space_complexity=state.current_space_complexity,
            is_optimal=False,
            is_novel=session.current_best_is_novel,
            correctness_proof=proof,
            why_this_works="",
            complexity_proof="",
            mathematical_essence=state.identified_structure or "unknown",
            actions_taken=session.actions_taken,
            solve_time_ms=solve_time_ms,
            iterations=len(session.actions_taken),
        )
