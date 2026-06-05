import pytest

from aalgoi.core.mind.cognitive_actions import ActionHandler, ActionParams, ActionResult, CognitiveAction


class TestCognitiveAction:
    def test_all_actions_have_unique_values(self):
        values = [int(a) for a in CognitiveAction]
        assert len(values) == len(set(values)), "Duplicate action values"
        assert len(values) == 25, f"Expected 25 actions, got {len(values)}"

    def test_action_names(self):
        assert CognitiveAction.DECOMPOSE_PROBLEM == 0
        assert CognitiveAction.SELECT_ALGORITHM == 10
        assert CognitiveAction.ACCEPT_SOLUTION == 22
        assert CognitiveAction.REQUEST_SIMPLIFY == 24

    def test_action_params_defaults(self):
        params = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        assert params.algorithm_id is None
        assert params.extra == {}

    def test_action_params_custom(self):
        params = ActionParams(
            action=CognitiveAction.SELECT_ALGORITHM,
            algorithm_id="quicksort",
        )
        assert params.algorithm_id == "quicksort"

    def test_action_result_defaults(self):
        result = ActionResult(
            action=CognitiveAction.SELECT_ALGORITHM,
            success=True,
            output="ok",
            solution_code="def f(): pass",
            correctness=0.9,
            time_complexity="O(n)",
            space_complexity="O(1)",
        )
        assert not result.is_novel_algorithm
        assert result.reward_signal == 0.0
        assert result.error is None


class TestActionHandler:
    @pytest.fixture
    def handler(self):
        class MockKG:
            def query_similar_problems(self, sig, top_k=5): return []
            def get_best_algorithms_for(self, sig, c): return []
            def get_known_failures(self, sig): return []
            def get_algorithm_code(self, n): return None
            def record_failure(self, a, s, r): pass
            def record_new_algorithm(self, a, s, acts): pass

        class MockExecutor:
            def execute(self, name, data): return None
            def execute_code(self, code, data): return None

        class MockSynth:
            def synthesize_novel(self, **kw): return None
            def apply_optimization(self, code, t, data): return None

        class MockProver:
            def prove(self, **kw):
                from aalgoi.core.reasoning.correctness_prover import CorrectnessProof
                return CorrectnessProof(is_correct=False, proof_type="mock",
                                        proof_text="", confidence=0.0, is_formal=False)
            def stress_test(self, **kw):
                from aalgoi.core.reasoning.correctness_prover import CorrectnessProof
                return CorrectnessProof(is_correct=False, proof_type="mock",
                                        proof_text="", confidence=0.0, is_formal=False)

        class MockComprehension:
            def comprehend(self, text, data):
                from aalgoi.core.reasoning.essence import ProblemEssence
                return ProblemEssence()
            def _parse_constraints(self, text): return {}

        return ActionHandler(
            kg=MockKG(),
            executor=MockExecutor(),
            synthesizer=MockSynth(),
            prover=MockProver(),
            comprehension=MockComprehension(),
        )

    def test_all_25_actions_have_handlers(self, handler):
        for action in CognitiveAction:
            assert action in handler._handlers, f"Missing handler for {action.name}"
            assert callable(handler._handlers[action]), f"Handler for {action.name} not callable"

    def test_dispatch_unknown_action(self, handler):
        import torch

        from aalgoi.core.mind.mind_state import MindState
        from aalgoi.core.mind.solving_loop import ThinkingSession

        state = MindState(problem_text="test", problem_signature="abc",
                          data_features=torch.zeros(64))
        session = ThinkingSession(problem_text="test", data=None,
                                  problem_signature="abc")

        result = handler.dispatch(
            CognitiveAction.BACKTRACK,
            ActionParams(action=CognitiveAction.BACKTRACK),
            state, session,
        )
        assert result.success is True

    def test_dispatch_all_without_error(self, handler):
        import torch

        from aalgoi.core.mind.mind_state import MindState
        from aalgoi.core.mind.solving_loop import ThinkingSession

        state = MindState(problem_text="sort this", problem_signature="abc",
                          data_features=torch.zeros(64))
        session = ThinkingSession(problem_text="sort this", data=[3, 1, 2],
                                  problem_signature="abc")

        for action in CognitiveAction:
            params = ActionParams(action=action)
            result = handler.dispatch(action, params, state, session)
            assert result is not None, f"No result for {action.name}"
