import pytest
import torch
from aalgoi.core.mind.solving_loop import MindSolvingLoop, ThinkingSession, UniversalSolution
from aalgoi.core.mind.rl_mind import AlgorithmicMind
from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph
from aalgoi.core.mind.safety_manager import MindSafetyManager
from aalgoi.core.mind.cognitive_actions import ActionHandler
from pathlib import Path


class TestThinkingSession:
    def test_initial_values(self):
        session = ThinkingSession(problem_text="test", data=None,
                                  problem_signature="abc")
        assert session.actions_taken == []
        assert session.current_best_code is None
        assert session.start_time > 0

    def test_time_limit_seconds_default(self):
        session = ThinkingSession(problem_text="test", data=None,
                                  problem_signature="abc")
        assert session.time_limit_seconds == 30.0

    def test_max_iterations_default(self):
        session = ThinkingSession(problem_text="test", data=None,
                                  problem_signature="abc")
        assert session.max_iterations == 50


class TestMindSolvingLoop:
    @pytest.fixture
    def loop(self):
        mind = AlgorithmicMind()
        kg = AlgorithmicKnowledgeGraph(Path.home() / ".aalgoi" / "kg_test")
        safety = MindSafetyManager(Path.home() / ".aalgoi")

        class MockExecutor:
            def execute(self, name, data): return None
            def execute_code(self, code, data): return None

        class MockSynth:
            def synthesize_novel(self, **kw): return None
            def apply_optimization(self, c, t, d): return None

        class MockProver:
            def prove(self, **kw):
                from aalgoi.core.reasoning.correctness_prover import CorrectnessProof
                return CorrectnessProof(True, "mock", "", 1.0, False)
            def stress_test(self, **kw):
                from aalgoi.core.reasoning.correctness_prover import CorrectnessProof
                return CorrectnessProof(True, "mock", "", 1.0, False)

        class MockComprehension:
            def comprehend(self, t, d):
                from aalgoi.core.reasoning.essence import ProblemEssence
                return ProblemEssence()
            def _parse_constraints(self, text): return {}

        handler = ActionHandler(
            kg=kg,
            executor=MockExecutor(),
            synthesizer=MockSynth(),
            prover=MockProver(),
            comprehension=MockComprehension(),
        )

        return MindSolvingLoop(mind=mind, kg=kg, action_handler=handler, safety=safety)

    def test_solve_returns_universal_solution(self, loop):
        solution = loop.solve("sort this array", [3, 1, 2], max_iterations=5)
        assert isinstance(solution, UniversalSolution)
        assert solution.solve_time_ms >= 0
        assert solution.iterations >= 0

    def test_solve_with_examples(self, loop):
        solution = loop.solve(
            "sort this array",
            [3, 1, 2],
            examples=[{"input": [3, 1, 2], "expected_output": [1, 2, 3]}],
            max_iterations=5,
        )
        assert isinstance(solution, UniversalSolution)

    def test_solve_time_is_measured(self, loop):
        solution = loop.solve("test", [1], max_iterations=5)
        assert solution.solve_time_ms > 0

    def test_get_available_actions_early_steps(self, loop):
        from aalgoi.core.mind.cognitive_actions import CognitiveAction

        session = ThinkingSession(problem_text="test", data=None,
                                  problem_signature="abc")
        state = loop._init_state(session)

        available = loop._get_available_actions(state, session)
        assert CognitiveAction.DECOMPOSE_PROBLEM in available
        assert CognitiveAction.IDENTIFY_STRUCTURE in available
        assert CognitiveAction.SELECT_ALGORITHM not in available

    def test_get_available_actions_after_some_steps(self, loop):
        from aalgoi.core.mind.cognitive_actions import CognitiveAction
        import torch
        from aalgoi.core.mind.mind_state import MindState

        state = MindState(problem_text="test", problem_signature="abc",
                          data_features=torch.zeros(64))
        session = ThinkingSession(problem_text="test", data=None,
                                  problem_signature="abc")
        session.actions_taken = [
            CognitiveAction.IDENTIFY_STRUCTURE,
            CognitiveAction.QUERY_SIMILAR,
            CognitiveAction.QUERY_PRINCIPLE,
            CognitiveAction.QUERY_ALGORITHMS,
        ]

        available = loop._get_available_actions(state, session)
        assert CognitiveAction.SELECT_ALGORITHM in available

    def test_best_complexity_comparison(self, loop):
        assert loop._is_better_complexity("O(n)", "O(n^2)")
        assert loop._is_better_complexity("O(log n)", "O(n)")
        assert not loop._is_better_complexity("O(n^2)", "O(n log n)")
