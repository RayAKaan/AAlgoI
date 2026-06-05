from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aalgoi.core.mind.adapters.executor_adapter import ExecutorAdapter
    from aalgoi.core.mind.adapters.prover_adapter import ProverAdapter
    from aalgoi.core.mind.adapters.synthesizer_adapter import SynthesizerAdapter
    from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph
    from aalgoi.core.mind.mind_state import MindState
    from aalgoi.core.mind.solving_loop import ThinkingSession
    from aalgoi.core.reasoning.comprehension_engine import DeepComprehensionEngine


class CognitiveAction(IntEnum):
    DECOMPOSE_PROBLEM = 0
    IDENTIFY_STRUCTURE = 1
    EXTRACT_CONSTRAINTS = 2
    FIND_INVARIANT = 3
    ESTIMATE_COMPLEXITY = 4

    QUERY_SIMILAR = 5
    QUERY_PRINCIPLE = 6
    QUERY_ALGORITHMS = 7
    QUERY_FAILURES = 8
    QUERY_COMPLEXITY = 9

    SELECT_ALGORITHM = 10
    MODIFY_ALGORITHM = 11
    COMBINE_ALGORITHMS = 12
    APPLY_OPTIMIZATION = 13
    SYNTHESIZE_NEW = 14
    DECOMPOSE_RECURSIVE = 15

    TEST_EXAMPLES = 16
    STRESS_TEST = 17
    CHECK_EDGE_CASES = 18
    VERIFY_COMPLEXITY = 19
    PROVE_CORRECTNESS = 20

    BACKTRACK = 21
    ACCEPT_SOLUTION = 22
    RECORD_DISCOVERY = 23
    REQUEST_SIMPLIFY = 24


@dataclass
class ActionParams:
    action: CognitiveAction
    algorithm_id: str | None = None
    modification_type: str | None = None
    optimization_type: str | None = None
    principle_name: str | None = None
    complexity_target: str | None = None
    extra: dict | None = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class ActionResult:
    action: CognitiveAction
    success: bool
    output: Any
    solution_code: str | None
    correctness: float
    time_complexity: str | None
    space_complexity: str | None
    is_novel_algorithm: bool = False
    reward_signal: float = 0.0
    error: str | None = None


class ActionHandler:
    def __init__(
        self,
        kg: "AlgorithmicKnowledgeGraph",
        executor: "ExecutorAdapter",
        synthesizer: "SynthesizerAdapter",
        prover: "ProverAdapter",
        comprehension: "DeepComprehensionEngine",
    ) -> None:
        self.kg = kg
        self.executor = executor
        self.synthesizer = synthesizer
        self.prover = prover
        self.comprehension = comprehension

        self._handlers: dict[CognitiveAction, Callable] = {
            CognitiveAction.DECOMPOSE_PROBLEM: self._handle_decompose,
            CognitiveAction.IDENTIFY_STRUCTURE: self._handle_identify_structure,
            CognitiveAction.EXTRACT_CONSTRAINTS: self._handle_extract_constraints,
            CognitiveAction.FIND_INVARIANT: self._handle_find_invariant,
            CognitiveAction.ESTIMATE_COMPLEXITY: self._handle_estimate_complexity,
            CognitiveAction.QUERY_SIMILAR: self._handle_query_similar,
            CognitiveAction.QUERY_PRINCIPLE: self._handle_query_principle,
            CognitiveAction.QUERY_ALGORITHMS: self._handle_query_algorithms,
            CognitiveAction.QUERY_FAILURES: self._handle_query_failures,
            CognitiveAction.QUERY_COMPLEXITY: self._handle_query_complexity,
            CognitiveAction.SELECT_ALGORITHM: self._handle_select_algorithm,
            CognitiveAction.MODIFY_ALGORITHM: self._handle_modify_algorithm,
            CognitiveAction.COMBINE_ALGORITHMS: self._handle_combine_algorithms,
            CognitiveAction.APPLY_OPTIMIZATION: self._handle_apply_optimization,
            CognitiveAction.SYNTHESIZE_NEW: self._handle_synthesize_new,
            CognitiveAction.DECOMPOSE_RECURSIVE: self._handle_decompose_recursive,
            CognitiveAction.TEST_EXAMPLES: self._handle_test_examples,
            CognitiveAction.STRESS_TEST: self._handle_stress_test,
            CognitiveAction.CHECK_EDGE_CASES: self._handle_edge_cases,
            CognitiveAction.VERIFY_COMPLEXITY: self._handle_verify_complexity,
            CognitiveAction.PROVE_CORRECTNESS: self._handle_prove_correctness,
            CognitiveAction.BACKTRACK: self._handle_backtrack,
            CognitiveAction.ACCEPT_SOLUTION: self._handle_accept,
            CognitiveAction.RECORD_DISCOVERY: self._handle_record_discovery,
            CognitiveAction.REQUEST_SIMPLIFY: self._handle_simplify,
        }

    def dispatch(
        self,
        action: CognitiveAction,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        handler = self._handlers.get(action)
        if handler is None:
            return ActionResult(
                action=action,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error=f"No handler for action {action.name}",
            )
        try:
            return handler(params, state, session)
        except Exception as e:
            return ActionResult(
                action=action,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error=str(e),
            )

    def _handle_identify_structure(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        essence = self.comprehension.comprehend(
            session.problem_text, session.data
        )
        return ActionResult(
            action=CognitiveAction.IDENTIFY_STRUCTURE,
            success=essence.hidden_structure != "unknown",
            output=essence,
            solution_code=None,
            correctness=0.0,
            time_complexity=essence.time_budget,
            space_complexity=None,
            reward_signal=0.3 if essence.hidden_structure != "unknown" else 0.0,
        )

    def _handle_decompose(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.DECOMPOSE_PROBLEM,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_extract_constraints(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        constraints = self.comprehension._parse_constraints(session.problem_text)
        time_budget = self.comprehension._derive_time_budget(
            constraints.get("n", 10**5)
        )
        return ActionResult(
            action=CognitiveAction.EXTRACT_CONSTRAINTS,
            success=bool(constraints),
            output={"constraints": constraints, "time_budget": time_budget},
            solution_code=None,
            correctness=0.0,
            time_complexity=time_budget,
            space_complexity=None,
            reward_signal=0.1,
        )

    def _handle_find_invariant(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.FIND_INVARIANT,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_estimate_complexity(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.ESTIMATE_COMPLEXITY,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_query_similar(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        sig = session.problem_signature
        similar = self.kg.query_similar_problems(sig, top_k=5)
        return ActionResult(
            action=CognitiveAction.QUERY_SIMILAR,
            success=len(similar) > 0,
            output=similar,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.1 * min(len(similar), 3),
        )

    def _handle_query_principle(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.QUERY_PRINCIPLE,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_query_algorithms(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        candidates = self.kg.get_best_algorithms_for(
            session.problem_signature,
            state.constraint_profile,
        )
        return ActionResult(
            action=CognitiveAction.QUERY_ALGORITHMS,
            success=len(candidates) > 0,
            output=candidates,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.05,
        )

    def _handle_query_failures(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        failed_algos = self.kg.get_known_failures(session.problem_signature)
        return ActionResult(
            action=CognitiveAction.QUERY_FAILURES,
            success=True,
            output={"known_failures": failed_algos},
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.05,
        )

    def _handle_query_complexity(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.QUERY_COMPLEXITY,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_select_algorithm(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        algo_id = params.algorithm_id
        if not algo_id:
            candidates = self.kg.get_best_algorithms_for(
                session.problem_signature,
                state.constraint_profile,
            )
            if not candidates:
                return ActionResult(
                    action=CognitiveAction.SELECT_ALGORITHM,
                    success=False,
                    output=None,
                    solution_code=None,
                    correctness=0.0,
                    time_complexity=None,
                    space_complexity=None,
                    error="No candidates in KG for this problem",
                )
            algo = candidates[0]
            algo_id = algo.name if hasattr(algo, "name") else str(algo)

        result = self.executor.execute(algo_id, session.data)
        correctness = self._quick_verify(result, session.examples)

        return ActionResult(
            action=CognitiveAction.SELECT_ALGORITHM,
            success=result is not None,
            output=result,
            solution_code=self.kg.get_algorithm_code(algo_id),
            correctness=correctness,
            time_complexity=None,
            space_complexity=None,
            reward_signal=correctness * 0.5,
        )

    def _handle_modify_algorithm(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.MODIFY_ALGORITHM,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_combine_algorithms(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.COMBINE_ALGORITHMS,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_apply_optimization(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        if not session.current_best_code:
            return ActionResult(
                action=CognitiveAction.APPLY_OPTIMIZATION,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="No current solution to optimize",
            )

        opt_type = params.optimization_type or "memoize"
        optimized_code = self.synthesizer.apply_optimization(
            session.current_best_code,
            opt_type,
            session.data,
        )

        if not optimized_code:
            return ActionResult(
                action=CognitiveAction.APPLY_OPTIMIZATION,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="Optimization stub returned None",
            )

        result = self.executor.execute_code(optimized_code, session.data)
        correctness = self._quick_verify(result, session.examples)

        return ActionResult(
            action=CognitiveAction.APPLY_OPTIMIZATION,
            success=correctness > 0.8,
            output=result,
            solution_code=optimized_code,
            correctness=correctness,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.3 if correctness > 0.8 else -0.1,
        )

    def _handle_synthesize_new(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        principle = params.principle_name or state.identified_principle
        complexity_target = params.complexity_target or state.target_complexity

        new_code = self.synthesizer.synthesize_novel(
            problem_text=session.problem_text,
            data=session.data,
            principle=principle,
            complexity_target=complexity_target,
            failed_approaches=session.failed_approaches,
            kg_context=state.kg_neighborhood,
        )

        if not new_code:
            return ActionResult(
                action=CognitiveAction.SYNTHESIZE_NEW,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="Synthesis stub returned None",
            )

        result = self.executor.execute_code(new_code, session.data)
        correctness = self._quick_verify(result, session.examples)

        return ActionResult(
            action=CognitiveAction.SYNTHESIZE_NEW,
            success=correctness > 0.5,
            output=result,
            solution_code=new_code,
            correctness=correctness,
            time_complexity=None,
            space_complexity=None,
            is_novel_algorithm=True,
            reward_signal=correctness * 2.0,
        )

    def _handle_decompose_recursive(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.DECOMPOSE_RECURSIVE,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_test_examples(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        if not session.current_best_code:
            return ActionResult(
                action=CognitiveAction.TEST_EXAMPLES,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="No solution to test",
            )

        correctness = self._quick_verify(
            self.executor.execute_code(session.current_best_code, session.data),
            session.examples,
        )

        return ActionResult(
            action=CognitiveAction.TEST_EXAMPLES,
            success=correctness > 0.8,
            output=None,
            solution_code=session.current_best_code,
            correctness=correctness,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.3 if correctness > 0.8 else -0.1,
        )

    def _handle_stress_test(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        if not session.current_best_code:
            return ActionResult(
                action=CognitiveAction.STRESS_TEST,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="No solution to stress test",
            )

        proof = self.prover.stress_test(
            candidate_code=session.current_best_code,
            brute_force_code=None,
            essence=state.problem_essence,
            iterations=1000,
        )

        return ActionResult(
            action=CognitiveAction.STRESS_TEST,
            success=proof.is_correct,
            output=proof,
            solution_code=session.current_best_code,
            correctness=proof.confidence,
            time_complexity=None,
            space_complexity=None,
            reward_signal=1.0 if proof.is_correct else -0.5,
        )

    def _handle_edge_cases(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.CHECK_EDGE_CASES,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_verify_complexity(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.VERIFY_COMPLEXITY,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _handle_prove_correctness(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        if not session.current_best_code:
            return ActionResult(
                action=CognitiveAction.PROVE_CORRECTNESS,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="No solution to prove",
            )

        proof = self.prover.prove(
            candidate_code=session.current_best_code,
            problem_text=session.problem_text,
            data=session.data,
            examples=session.examples,
            essence=state.problem_essence,
        )

        return ActionResult(
            action=CognitiveAction.PROVE_CORRECTNESS,
            success=proof.is_correct,
            output=proof,
            solution_code=session.current_best_code,
            correctness=proof.confidence,
            time_complexity=None,
            space_complexity=None,
            reward_signal=1.5 if proof.is_correct and proof.is_formal
            else 0.8 if proof.is_correct
            else -0.3,
        )

    def _handle_backtrack(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        if session.current_best_code and session.current_best_algo:
            self.kg.record_failure(
                algorithm_name=session.current_best_algo,
                problem_signature=session.problem_signature,
                failure_reason="backtracked_by_mind",
            )
            if session.current_best_algo not in session.failed_approaches:
                session.failed_approaches.append(session.current_best_algo)
            session.current_best_code = None
            session.current_best_algo = None

        return ActionResult(
            action=CognitiveAction.BACKTRACK,
            success=True,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=-0.2,
        )

    def _handle_accept(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.ACCEPT_SOLUTION,
            success=session.current_best_code is not None,
            output=None,
            solution_code=session.current_best_code,
            correctness=state.correctness_confidence,
            time_complexity=state.current_complexity,
            space_complexity=state.current_space_complexity,
            reward_signal=1.0 if session.current_best_code else 0.0,
        )

    def _handle_record_discovery(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        if not (session.current_best_code
                and session.current_best_verified
                and session.current_best_is_novel):
            return ActionResult(
                action=CognitiveAction.RECORD_DISCOVERY,
                success=False,
                output=None,
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
                error="Nothing verified+novel to record",
            )

        import hashlib
        algo_name = (
            f"discovered_"
            f"{hashlib.sha256(session.current_best_code.encode()).hexdigest()[:8]}"
        )

        self.kg.record_new_algorithm(
            algo_name,
            session.problem_signature,
            session.actions_taken,
        )

        return ActionResult(
            action=CognitiveAction.RECORD_DISCOVERY,
            success=True,
            output={"algorithm_name": algo_name},
            solution_code=session.current_best_code,
            correctness=state.correctness_confidence,
            time_complexity=state.current_complexity,
            space_complexity=state.current_space_complexity,
            reward_signal=3.0,
        )

    def _handle_simplify(
        self,
        params: ActionParams,
        state: "MindState",
        session: "ThinkingSession",
    ) -> ActionResult:
        return ActionResult(
            action=CognitiveAction.REQUEST_SIMPLIFY,
            success=False,
            output=None,
            solution_code=None,
            correctness=0.0,
            time_complexity=None,
            space_complexity=None,
            reward_signal=0.0,
        )

    def _quick_verify(self, result: Any, examples: list[dict]) -> float:
        if not examples or result is None:
            return 0.5
        passed = sum(1 for ex in examples if result == ex.get("expected_output"))
        return passed / len(examples)

    def _is_better_complexity(self, new: str, old: str) -> bool:
        order = [
            "O(1)", "O(log n)", "O(n)", "O(n log n)",
            "O(n^2)", "O(2^n)", "unknown",
        ]
        try:
            return order.index(new) < order.index(old)
        except ValueError:
            return False
