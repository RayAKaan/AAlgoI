import hashlib
from unittest.mock import MagicMock, PropertyMock

import pytest

from aalgoi.core.mind.cognitive_actions import (
    ActionHandler,
    ActionParams,
    ActionResult,
    CognitiveAction,
)


class TestCognitiveActionEnum:
    def test_25_members(self):
        assert len(CognitiveAction) == 25

    def test_all_values_are_unique(self):
        values = [int(m) for m in CognitiveAction]
        assert len(values) == len(set(values))

    def test_core_enum_values(self):
        assert CognitiveAction.DECOMPOSE_PROBLEM == 0
        assert CognitiveAction.IDENTIFY_STRUCTURE == 1
        assert CognitiveAction.EXTRACT_CONSTRAINTS == 2
        assert CognitiveAction.FIND_INVARIANT == 3
        assert CognitiveAction.ESTIMATE_COMPLEXITY == 4
        assert CognitiveAction.QUERY_SIMILAR == 5
        assert CognitiveAction.QUERY_PRINCIPLE == 6
        assert CognitiveAction.QUERY_ALGORITHMS == 7
        assert CognitiveAction.QUERY_FAILURES == 8
        assert CognitiveAction.QUERY_COMPLEXITY == 9
        assert CognitiveAction.SELECT_ALGORITHM == 10
        assert CognitiveAction.MODIFY_ALGORITHM == 11
        assert CognitiveAction.COMBINE_ALGORITHMS == 12
        assert CognitiveAction.APPLY_OPTIMIZATION == 13
        assert CognitiveAction.SYNTHESIZE_NEW == 14
        assert CognitiveAction.DECOMPOSE_RECURSIVE == 15
        assert CognitiveAction.TEST_EXAMPLES == 16
        assert CognitiveAction.STRESS_TEST == 17
        assert CognitiveAction.CHECK_EDGE_CASES == 18
        assert CognitiveAction.VERIFY_COMPLEXITY == 19
        assert CognitiveAction.PROVE_CORRECTNESS == 20
        assert CognitiveAction.BACKTRACK == 21
        assert CognitiveAction.ACCEPT_SOLUTION == 22
        assert CognitiveAction.RECORD_DISCOVERY == 23
        assert CognitiveAction.REQUEST_SIMPLIFY == 24

    def test_is_intenum(self):
        assert isinstance(CognitiveAction.DECOMPOSE_PROBLEM, int)
        assert int(CognitiveAction.ACCEPT_SOLUTION) == 22

    def test_names_are_consistent(self):
        names = [m.name for m in CognitiveAction]
        assert "DECOMPOSE_PROBLEM" in names
        assert "ACCEPT_SOLUTION" in names
        assert "REQUEST_SIMPLIFY" in names


class TestActionParams:
    def test_defaults(self):
        p = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        assert p.action == CognitiveAction.SELECT_ALGORITHM
        assert p.algorithm_id is None
        assert p.modification_type is None
        assert p.optimization_type is None
        assert p.principle_name is None
        assert p.complexity_target is None
        assert p.extra == {}

    def test_extra_defaults_to_empty_dict_when_none(self):
        p = ActionParams(action=CognitiveAction.SELECT_ALGORITHM, extra=None)
        assert p.extra == {}

    def test_extra_preserves_dict_value(self):
        p = ActionParams(action=CognitiveAction.SELECT_ALGORITHM, extra={"key": "val"})
        assert p.extra == {"key": "val"}

    def test_all_fields(self):
        p = ActionParams(
            action=CognitiveAction.APPLY_OPTIMIZATION,
            algorithm_id="merge_sort",
            modification_type="parallel",
            optimization_type="memoize",
            principle_name="divide_and_conquer",
            complexity_target="O(n log n)",
            extra={"iterations": 3},
        )
        assert p.algorithm_id == "merge_sort"
        assert p.modification_type == "parallel"
        assert p.optimization_type == "memoize"
        assert p.principle_name == "divide_and_conquer"
        assert p.complexity_target == "O(n log n)"
        assert p.extra == {"iterations": 3}

    def test_extra_mutable_default(self):
        p1 = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        p2 = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        assert p1.extra is not p2.extra
        p1.extra["a"] = 1
        assert "a" not in p2.extra


class TestActionResult:
    @pytest.fixture
    def base(self):
        return ActionResult(
            action=CognitiveAction.DECOMPOSE_PROBLEM,
            success=True,
            output="out",
            solution_code="code",
            correctness=1.0,
            time_complexity="O(n)",
            space_complexity="O(1)",
        )

    def test_defaults(self, base):
        assert base.is_novel_algorithm is False
        assert base.reward_signal == 0.0
        assert base.error is None

    def test_custom_defaults(self):
        r = ActionResult(
            action=CognitiveAction.SYNTHESIZE_NEW,
            success=True,
            output="out",
            solution_code="code",
            correctness=0.9,
            time_complexity=None,
            space_complexity=None,
            is_novel_algorithm=True,
            reward_signal=2.0,
            error="something",
        )
        assert r.is_novel_algorithm is True
        assert r.reward_signal == 2.0
        assert r.error == "something"

    def test_mutable_fields(self, base):
        base.is_novel_algorithm = True
        base.error = "err"
        assert base.is_novel_algorithm
        assert base.error == "err"


class TestActionHandlerInit:
    def test_constructor_stores_dependencies(self):
        kg = MagicMock()
        executor = MagicMock()
        synthesizer = MagicMock()
        prover = MagicMock()
        comprehension = MagicMock()
        handler = ActionHandler(kg, executor, synthesizer, prover, comprehension)
        assert handler.kg is kg
        assert handler.executor is executor
        assert handler.synthesizer is synthesizer
        assert handler.prover is prover
        assert handler.comprehension is comprehension

    def test_all_25_handlers_registered(self):
        kg = MagicMock()
        handler = ActionHandler(kg, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert len(handler._handlers) == 25
        for action in CognitiveAction:
            assert action in handler._handlers
            assert callable(handler._handlers[action])


class TestActionHandlerDispatch:
    @pytest.fixture
    def handler(self):
        return ActionHandler(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

    @pytest.fixture
    def state_session(self):
        state = MagicMock()
        session = MagicMock()
        return state, session

    def test_dispatch_unknown_action(self, handler, state_session):
        state, session = state_session
        handler._handlers = {}
        params = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        result = handler.dispatch(CognitiveAction.SELECT_ALGORITHM, params, state, session)
        assert result.success is False
        assert result.output is None
        assert result.error == "No handler for action SELECT_ALGORITHM"
        assert result.action == CognitiveAction.SELECT_ALGORITHM

    def test_dispatch_catches_exception(self, handler, state_session):
        state, session = state_session
        handler._handlers = {
            CognitiveAction.DECOMPOSE_PROBLEM: lambda p, s, sess: (_ for _ in ()).throw(
                ValueError("fail")
            ),
        }
        params = ActionParams(action=CognitiveAction.DECOMPOSE_PROBLEM)
        result = handler.dispatch(CognitiveAction.DECOMPOSE_PROBLEM, params, state, session)
        assert result.success is False
        assert result.error == "fail"
        assert result.action == CognitiveAction.DECOMPOSE_PROBLEM

    def test_dispatch_routes_correctly(self, handler, state_session):
        state, session = state_session
        captured = []

        def my_handler(p, s, sess):
            captured.append((p, s, sess))
            return ActionResult(
                action=CognitiveAction.DECOMPOSE_PROBLEM,
                success=True,
                output="ok",
                solution_code=None,
                correctness=0.0,
                time_complexity=None,
                space_complexity=None,
            )

        handler._handlers = {CognitiveAction.DECOMPOSE_PROBLEM: my_handler}
        params = ActionParams(action=CognitiveAction.DECOMPOSE_PROBLEM)
        result = handler.dispatch(CognitiveAction.DECOMPOSE_PROBLEM, params, state, session)
        assert result.success
        assert result.output == "ok"
        assert len(captured) == 1
        assert captured[0][0] is params
        assert captured[0][1] is state
        assert captured[0][2] is session


class TestHandlers:
    @pytest.fixture
    def handler(self):
        return ActionHandler(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

    @pytest.fixture
    def state(self):
        return MagicMock()

    @pytest.fixture
    def session(self):
        s = MagicMock()
        s.failed_approaches = []
        s.actions_taken = []
        return s

    # ---- Simple stub handlers (no complex dependencies) ----

    def test_handle_decompose(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.DECOMPOSE_PROBLEM)
        result = handler.dispatch(CognitiveAction.DECOMPOSE_PROBLEM, params, state, session)
        assert result.action == CognitiveAction.DECOMPOSE_PROBLEM
        assert result.success is False
        assert result.output is None
        assert result.reward_signal == 0.0

    def test_handle_find_invariant(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.FIND_INVARIANT)
        result = handler.dispatch(CognitiveAction.FIND_INVARIANT, params, state, session)
        assert result.action == CognitiveAction.FIND_INVARIANT
        assert result.success is False

    def test_handle_estimate_complexity(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.ESTIMATE_COMPLEXITY)
        result = handler.dispatch(CognitiveAction.ESTIMATE_COMPLEXITY, params, state, session)
        assert result.action == CognitiveAction.ESTIMATE_COMPLEXITY
        assert result.success is False

    def test_handle_query_principle(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.QUERY_PRINCIPLE)
        result = handler.dispatch(CognitiveAction.QUERY_PRINCIPLE, params, state, session)
        assert result.action == CognitiveAction.QUERY_PRINCIPLE
        assert result.success is False

    def test_handle_query_complexity(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.QUERY_COMPLEXITY)
        result = handler.dispatch(CognitiveAction.QUERY_COMPLEXITY, params, state, session)
        assert result.action == CognitiveAction.QUERY_COMPLEXITY
        assert result.success is False

    def test_handle_modify_algorithm(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.MODIFY_ALGORITHM)
        result = handler.dispatch(CognitiveAction.MODIFY_ALGORITHM, params, state, session)
        assert result.action == CognitiveAction.MODIFY_ALGORITHM
        assert result.success is False

    def test_handle_combine_algorithms(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.COMBINE_ALGORITHMS)
        result = handler.dispatch(CognitiveAction.COMBINE_ALGORITHMS, params, state, session)
        assert result.action == CognitiveAction.COMBINE_ALGORITHMS
        assert result.success is False

    def test_handle_decompose_recursive(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.DECOMPOSE_RECURSIVE)
        result = handler.dispatch(CognitiveAction.DECOMPOSE_RECURSIVE, params, state, session)
        assert result.action == CognitiveAction.DECOMPOSE_RECURSIVE
        assert result.success is False

    def test_handle_edge_cases(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.CHECK_EDGE_CASES)
        result = handler.dispatch(CognitiveAction.CHECK_EDGE_CASES, params, state, session)
        assert result.action == CognitiveAction.CHECK_EDGE_CASES
        assert result.success is False

    def test_handle_verify_complexity(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.VERIFY_COMPLEXITY)
        result = handler.dispatch(CognitiveAction.VERIFY_COMPLEXITY, params, state, session)
        assert result.action == CognitiveAction.VERIFY_COMPLEXITY
        assert result.success is False

    def test_handle_simplify(self, handler, state, session):
        params = ActionParams(action=CognitiveAction.REQUEST_SIMPLIFY)
        result = handler.dispatch(CognitiveAction.REQUEST_SIMPLIFY, params, state, session)
        assert result.action == CognitiveAction.REQUEST_SIMPLIFY
        assert result.success is False

    # ---- Handlers with external dependencies ----

    def test_handle_identify_structure_success(self, handler, state, session):
        essence = MagicMock()
        essence.hidden_structure = "total_order"
        essence.time_budget = "O(n log n)"
        handler.comprehension.comprehend.return_value = essence
        session.problem_text = "sort this"
        session.data = [3, 1, 2]

        params = ActionParams(action=CognitiveAction.IDENTIFY_STRUCTURE)
        result = handler.dispatch(
            CognitiveAction.IDENTIFY_STRUCTURE, params, state, session,
        )

        handler.comprehension.comprehend.assert_called_once_with(
            session.problem_text, session.data,
        )
        assert result.success is True
        assert result.output is essence
        assert result.time_complexity == "O(n log n)"
        assert result.reward_signal == 0.3

    def test_handle_identify_structure_unknown_structure(self, handler, state, session):
        essence = MagicMock()
        essence.hidden_structure = "unknown"
        essence.time_budget = "unknown"
        handler.comprehension.comprehend.return_value = essence
        session.problem_text = "foo"
        session.data = None

        params = ActionParams(action=CognitiveAction.IDENTIFY_STRUCTURE)
        result = handler.dispatch(
            CognitiveAction.IDENTIFY_STRUCTURE, params, state, session,
        )

        assert result.success is False
        assert result.reward_signal == 0.0

    def test_handle_extract_constraints(self, handler, state, session):
        handler.comprehension._parse_constraints.return_value = {"n": 1000}
        handler.comprehension._derive_time_budget.return_value = "O(n^2)"
        session.problem_text = "sort n numbers"

        params = ActionParams(action=CognitiveAction.EXTRACT_CONSTRAINTS)
        result = handler.dispatch(
            CognitiveAction.EXTRACT_CONSTRAINTS, params, state, session,
        )

        handler.comprehension._parse_constraints.assert_called_once_with(
            session.problem_text,
        )
        assert result.success is True
        assert result.output == {"constraints": {"n": 1000}, "time_budget": "O(n^2)"}
        assert result.time_complexity == "O(n^2)"
        assert result.reward_signal == 0.1

    def test_handle_extract_constraints_empty(self, handler, state, session):
        handler.comprehension._parse_constraints.return_value = {}
        handler.comprehension._derive_time_budget.return_value = "O(n log n)"
        session.problem_text = ""

        params = ActionParams(action=CognitiveAction.EXTRACT_CONSTRAINTS)
        result = handler.dispatch(
            CognitiveAction.EXTRACT_CONSTRAINTS, params, state, session,
        )

        assert result.success is False

    def test_handle_query_similar_found(self, handler, state, session):
        similar = ["prob1", "prob2"]
        handler.kg.query_similar_problems.return_value = similar
        session.problem_signature = "sig123"

        params = ActionParams(action=CognitiveAction.QUERY_SIMILAR)
        result = handler.dispatch(
            CognitiveAction.QUERY_SIMILAR, params, state, session,
        )

        handler.kg.query_similar_problems.assert_called_once_with("sig123", top_k=5)
        assert result.success is True
        assert result.output == similar
        assert result.reward_signal == 0.2

    def test_handle_query_similar_empty(self, handler, state, session):
        handler.kg.query_similar_problems.return_value = []
        session.problem_signature = "sig456"

        params = ActionParams(action=CognitiveAction.QUERY_SIMILAR)
        result = handler.dispatch(
            CognitiveAction.QUERY_SIMILAR, params, state, session,
        )

        assert result.success is False
        assert result.output == []
        assert result.reward_signal == 0.0

    def test_handle_query_algorithms_found(self, handler, state, session):
        candidates = ["algo1", "algo2"]
        handler.kg.get_best_algorithms_for.return_value = candidates
        session.problem_signature = "sig"
        state.constraint_profile = {"n": 100}

        params = ActionParams(action=CognitiveAction.QUERY_ALGORITHMS)
        result = handler.dispatch(
            CognitiveAction.QUERY_ALGORITHMS, params, state, session,
        )

        handler.kg.get_best_algorithms_for.assert_called_once_with(
            "sig", {"n": 100},
        )
        assert result.success is True
        assert result.output == candidates
        assert result.reward_signal == 0.05

    def test_handle_query_algorithms_empty(self, handler, state, session):
        handler.kg.get_best_algorithms_for.return_value = []

        params = ActionParams(action=CognitiveAction.QUERY_ALGORITHMS)
        result = handler.dispatch(
            CognitiveAction.QUERY_ALGORITHMS, params, state, session,
        )

        assert result.success is False
        assert result.output == []

    def test_handle_query_failures(self, handler, state, session):
        handler.kg.get_known_failures.return_value = ["fail1"]
        session.problem_signature = "sig"

        params = ActionParams(action=CognitiveAction.QUERY_FAILURES)
        result = handler.dispatch(
            CognitiveAction.QUERY_FAILURES, params, state, session,
        )

        handler.kg.get_known_failures.assert_called_once_with("sig")
        assert result.success is True
        assert result.output == {"known_failures": ["fail1"]}
        assert result.reward_signal == 0.05

    # ---- SELECT_ALGORITHM ----

    def test_handle_select_algorithm_with_id(self, handler, state, session):
        params = ActionParams(
            action=CognitiveAction.SELECT_ALGORITHM, algorithm_id="quick_sort",
        )
        handler.kg.get_algorithm_code.return_value = "def quick_sort(): pass"
        handler.executor.execute.return_value = [1, 2, 3]
        session.data = [3, 1, 2]
        session.examples = [{"expected_output": [1, 2, 3]}]

        result = handler.dispatch(
            CognitiveAction.SELECT_ALGORITHM, params, state, session,
        )

        handler.executor.execute.assert_called_once_with("quick_sort", [3, 1, 2])
        assert result.success is True
        assert result.output == [1, 2, 3]
        assert result.correctness == 1.0

    def test_handle_select_algorithm_no_id_with_candidates(self, handler, state, session):
        candidate = MagicMock()
        candidate.name = "quick_sort"
        handler.kg.get_best_algorithms_for.return_value = [candidate]
        handler.kg.get_algorithm_code.return_value = "def quick_sort(): pass"
        handler.executor.execute.return_value = [1, 2, 3]
        session.problem_signature = "sig"
        state.constraint_profile = {"n": 100}
        session.data = [3, 1, 2]
        session.examples = [{"expected_output": [1, 2, 3]}]

        params = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        result = handler.dispatch(
            CognitiveAction.SELECT_ALGORITHM, params, state, session,
        )

        handler.kg.get_best_algorithms_for.assert_called_once_with("sig", {"n": 100})
        assert result.success is True
        assert result.correctness == 1.0

    def test_handle_select_algorithm_no_candidates(self, handler, state, session):
        handler.kg.get_best_algorithms_for.return_value = []
        session.problem_signature = "sig"
        state.constraint_profile = {}

        params = ActionParams(action=CognitiveAction.SELECT_ALGORITHM)
        result = handler.dispatch(
            CognitiveAction.SELECT_ALGORITHM, params, state, session,
        )

        assert result.success is False
        assert result.error == "No candidates in KG for this problem"

    def test_handle_select_algorithm_executor_returns_none(self, handler, state, session):
        handler.kg.get_algorithm_code.return_value = "def f(): pass"
        handler.executor.execute.return_value = None
        session.data = [1]
        session.examples = [{"expected_output": 1}]

        params = ActionParams(
            action=CognitiveAction.SELECT_ALGORITHM, algorithm_id="algo",
        )
        result = handler.dispatch(
            CognitiveAction.SELECT_ALGORITHM, params, state, session,
        )

        assert result.success is False
        assert result.correctness == 0.5

    # ---- APPLY_OPTIMIZATION ----

    def test_handle_apply_optimization_success(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        session.data = [3, 1, 2]
        session.examples = [{"expected_output": [1, 2, 3]}]
        handler.synthesizer.apply_optimization.return_value = "def solve_opt(): pass"
        handler.executor.execute_code.return_value = [1, 2, 3]

        params = ActionParams(
            action=CognitiveAction.APPLY_OPTIMIZATION, optimization_type="memoize",
        )
        result = handler.dispatch(
            CognitiveAction.APPLY_OPTIMIZATION, params, state, session,
        )

        handler.synthesizer.apply_optimization.assert_called_once_with(
            "def solve(): pass", "memoize", [3, 1, 2],
        )
        handler.executor.execute_code.assert_called_once_with(
            "def solve_opt(): pass", [3, 1, 2],
        )
        assert result.success is True
        assert result.solution_code == "def solve_opt(): pass"
        assert result.correctness == 1.0
        assert result.reward_signal == 0.3

    def test_handle_apply_optimization_no_code(self, handler, state, session):
        session.current_best_code = None

        params = ActionParams(action=CognitiveAction.APPLY_OPTIMIZATION)
        result = handler.dispatch(
            CognitiveAction.APPLY_OPTIMIZATION, params, state, session,
        )

        assert result.success is False
        assert result.error == "No current solution to optimize"

    def test_handle_apply_optimization_stub_returns_none(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        handler.synthesizer.apply_optimization.return_value = None

        params = ActionParams(action=CognitiveAction.APPLY_OPTIMIZATION)
        result = handler.dispatch(
            CognitiveAction.APPLY_OPTIMIZATION, params, state, session,
        )

        assert result.success is False
        assert result.error == "Optimization stub returned None"

    def test_handle_apply_optimization_low_correctness(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        session.data = [1]
        session.examples = [{"expected_output": 99}]
        handler.synthesizer.apply_optimization.return_value = "def opt(): pass"
        handler.executor.execute_code.return_value = 0

        params = ActionParams(action=CognitiveAction.APPLY_OPTIMIZATION)
        result = handler.dispatch(
            CognitiveAction.APPLY_OPTIMIZATION, params, state, session,
        )

        assert result.success is False
        assert result.reward_signal == -0.1

    # ---- SYNTHESIZE_NEW ----

    def test_handle_synthesize_new_success(self, handler, state, session):
        session.problem_text = "sort this"
        session.data = [3, 1, 2]
        session.failed_approaches = ["bubble"]
        state.kg_neighborhood = ["algo1"]
        state.identified_principle = ""
        state.target_complexity = ""
        session.examples = [{"expected_output": [1, 2, 3]}]
        handler.synthesizer.synthesize_novel.return_value = "def novel(): pass"
        handler.executor.execute_code.return_value = [1, 2, 3]

        params = ActionParams(
            action=CognitiveAction.SYNTHESIZE_NEW,
            principle_name="divide_and_conquer",
            complexity_target="O(n log n)",
        )
        result = handler.dispatch(
            CognitiveAction.SYNTHESIZE_NEW, params, state, session,
        )

        handler.synthesizer.synthesize_novel.assert_called_once_with(
            problem_text="sort this",
            data=[3, 1, 2],
            principle="divide_and_conquer",
            complexity_target="O(n log n)",
            failed_approaches=["bubble"],
            kg_context=["algo1"],
        )
        handler.executor.execute_code.assert_called_once_with(
            "def novel(): pass", [3, 1, 2],
        )
        assert result.success is True
        assert result.solution_code == "def novel(): pass"
        assert result.is_novel_algorithm is True
        assert result.correctness == 1.0
        assert result.reward_signal == 2.0

    def test_handle_synthesize_new_from_state_fallbacks(self, handler, state, session):
        session.problem_text = "text"
        session.data = None
        session.failed_approaches = []
        state.kg_neighborhood = []
        state.identified_principle = "greedy"
        state.target_complexity = "O(n)"
        session.examples = [{"expected_output": 42}]
        handler.synthesizer.synthesize_novel.return_value = "def f(): return 42"
        handler.executor.execute_code.return_value = 42

        params = ActionParams(
            action=CognitiveAction.SYNTHESIZE_NEW,
            principle_name=None,
            complexity_target=None,
        )
        result = handler.dispatch(
            CognitiveAction.SYNTHESIZE_NEW, params, state, session,
        )

        handler.synthesizer.synthesize_novel.assert_called_once()
        call_kwargs = handler.synthesizer.synthesize_novel.call_args[1]
        assert call_kwargs["principle"] == "greedy"
        assert call_kwargs["complexity_target"] == "O(n)"

    def test_handle_synthesize_new_returns_none(self, handler, state, session):
        handler.synthesizer.synthesize_novel.return_value = None
        session.problem_text = "text"
        session.data = None
        session.failed_approaches = []
        state.kg_neighborhood = []

        params = ActionParams(action=CognitiveAction.SYNTHESIZE_NEW)
        result = handler.dispatch(
            CognitiveAction.SYNTHESIZE_NEW, params, state, session,
        )

        assert result.success is False
        assert result.error == "Synthesis stub returned None"

    # ---- TEST_EXAMPLES ----

    def test_handle_test_examples_passes(self, handler, state, session):
        session.current_best_code = "def solve(): return 42"
        session.data = [1]
        session.examples = [{"expected_output": 42}]
        handler.executor.execute_code.return_value = 42

        params = ActionParams(action=CognitiveAction.TEST_EXAMPLES)
        result = handler.dispatch(
            CognitiveAction.TEST_EXAMPLES, params, state, session,
        )

        handler.executor.execute_code.assert_called_once_with(
            "def solve(): return 42", [1],
        )
        assert result.success is True
        assert result.correctness == 1.0
        assert result.reward_signal == 0.3

    def test_handle_test_examples_fails(self, handler, state, session):
        session.current_best_code = "def solve(): return 0"
        session.data = [1]
        session.examples = [{"expected_output": 42}]
        handler.executor.execute_code.return_value = 0

        params = ActionParams(action=CognitiveAction.TEST_EXAMPLES)
        result = handler.dispatch(
            CognitiveAction.TEST_EXAMPLES, params, state, session,
        )

        assert result.success is False
        assert result.reward_signal == -0.1

    def test_handle_test_examples_no_solution(self, handler, state, session):
        session.current_best_code = None

        params = ActionParams(action=CognitiveAction.TEST_EXAMPLES)
        result = handler.dispatch(
            CognitiveAction.TEST_EXAMPLES, params, state, session,
        )

        assert result.success is False
        assert result.error == "No solution to test"

    # ---- STRESS_TEST ----

    def test_handle_stress_test_passes(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        proof = MagicMock()
        proof.is_correct = True
        proof.confidence = 0.99
        handler.prover.stress_test.return_value = proof
        state.problem_essence = MagicMock()

        params = ActionParams(action=CognitiveAction.STRESS_TEST)
        result = handler.dispatch(
            CognitiveAction.STRESS_TEST, params, state, session,
        )

        handler.prover.stress_test.assert_called_once_with(
            candidate_code="def solve(): pass",
            brute_force_code=None,
            essence=state.problem_essence,
            iterations=1000,
        )
        assert result.success is True
        assert result.output is proof
        assert result.correctness == 0.99
        assert result.reward_signal == 1.0

    def test_handle_stress_test_fails(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        proof = MagicMock()
        proof.is_correct = False
        proof.confidence = 0.3
        handler.prover.stress_test.return_value = proof
        state.problem_essence = MagicMock()

        params = ActionParams(action=CognitiveAction.STRESS_TEST)
        result = handler.dispatch(
            CognitiveAction.STRESS_TEST, params, state, session,
        )

        assert result.success is False
        assert result.reward_signal == -0.5

    def test_handle_stress_test_no_solution(self, handler, state, session):
        session.current_best_code = None

        params = ActionParams(action=CognitiveAction.STRESS_TEST)
        result = handler.dispatch(
            CognitiveAction.STRESS_TEST, params, state, session,
        )

        assert result.success is False
        assert result.error == "No solution to stress test"

    # ---- PROVE_CORRECTNESS ----

    def test_handle_prove_correctness_formal(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        session.problem_text = "sort"
        session.data = [1]
        session.examples = [{"expected_output": 1}]
        state.problem_essence = MagicMock()
        proof = MagicMock()
        proof.is_correct = True
        proof.is_formal = True
        proof.confidence = 1.0
        handler.prover.prove.return_value = proof

        params = ActionParams(action=CognitiveAction.PROVE_CORRECTNESS)
        result = handler.dispatch(
            CognitiveAction.PROVE_CORRECTNESS, params, state, session,
        )

        handler.prover.prove.assert_called_once_with(
            candidate_code="def solve(): pass",
            problem_text="sort",
            data=[1],
            examples=[{"expected_output": 1}],
            essence=state.problem_essence,
        )
        assert result.success is True
        assert result.reward_signal == 1.5

    def test_handle_prove_correctness_empirical(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        session.problem_text = ""
        session.data = None
        session.examples = []
        state.problem_essence = MagicMock()
        proof = MagicMock()
        proof.is_correct = True
        proof.is_formal = False
        proof.confidence = 0.8
        handler.prover.prove.return_value = proof

        params = ActionParams(action=CognitiveAction.PROVE_CORRECTNESS)
        result = handler.dispatch(
            CognitiveAction.PROVE_CORRECTNESS, params, state, session,
        )

        assert result.success is True
        assert result.reward_signal == 0.8

    def test_handle_prove_correctness_fails(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        state.problem_essence = MagicMock()
        proof = MagicMock()
        proof.is_correct = False
        proof.confidence = 0.2
        handler.prover.prove.return_value = proof

        params = ActionParams(action=CognitiveAction.PROVE_CORRECTNESS)
        result = handler.dispatch(
            CognitiveAction.PROVE_CORRECTNESS, params, state, session,
        )

        assert result.success is False
        assert result.reward_signal == -0.3

    def test_handle_prove_correctness_no_solution(self, handler, state, session):
        session.current_best_code = None

        params = ActionParams(action=CognitiveAction.PROVE_CORRECTNESS)
        result = handler.dispatch(
            CognitiveAction.PROVE_CORRECTNESS, params, state, session,
        )

        assert result.success is False
        assert result.error == "No solution to prove"

    # ---- BACKTRACK ----

    def test_handle_backtrack_with_solution(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        session.current_best_algo = "quick_sort"
        session.problem_signature = "sig"

        params = ActionParams(action=CognitiveAction.BACKTRACK)
        result = handler.dispatch(
            CognitiveAction.BACKTRACK, params, state, session,
        )

        handler.kg.record_failure.assert_called_once_with(
            algorithm_name="quick_sort",
            problem_signature="sig",
            failure_reason="backtracked_by_mind",
        )
        assert "quick_sort" in session.failed_approaches
        assert session.current_best_code is None
        assert session.current_best_algo is None
        assert result.success is True
        assert result.reward_signal == -0.2

    def test_handle_backtrack_does_not_duplicate_failures(self, handler, state, session):
        session.current_best_code = "code"
        session.current_best_algo = "bubble"
        session.problem_signature = "sig"
        session.failed_approaches = ["bubble"]

        params = ActionParams(action=CognitiveAction.BACKTRACK)
        result = handler.dispatch(
            CognitiveAction.BACKTRACK, params, state, session,
        )

        handler.kg.record_failure.assert_called_once()
        assert session.failed_approaches == ["bubble"]

    def test_handle_backtrack_no_solution(self, handler, state, session):
        session.current_best_code = None
        session.current_best_algo = None

        params = ActionParams(action=CognitiveAction.BACKTRACK)
        result = handler.dispatch(
            CognitiveAction.BACKTRACK, params, state, session,
        )

        handler.kg.record_failure.assert_not_called()
        assert result.success is True
        assert result.reward_signal == -0.2

    # ---- ACCEPT_SOLUTION ----

    def test_handle_accept_with_solution(self, handler, state, session):
        session.current_best_code = "def solve(): pass"
        state.correctness_confidence = 0.95
        state.current_complexity = "O(n log n)"
        state.current_space_complexity = "O(n)"

        params = ActionParams(action=CognitiveAction.ACCEPT_SOLUTION)
        result = handler.dispatch(
            CognitiveAction.ACCEPT_SOLUTION, params, state, session,
        )

        assert result.success is True
        assert result.solution_code == "def solve(): pass"
        assert result.correctness == 0.95
        assert result.time_complexity == "O(n log n)"
        assert result.space_complexity == "O(n)"
        assert result.reward_signal == 1.0

    def test_handle_accept_without_solution(self, handler, state, session):
        session.current_best_code = None

        params = ActionParams(action=CognitiveAction.ACCEPT_SOLUTION)
        result = handler.dispatch(
            CognitiveAction.ACCEPT_SOLUTION, params, state, session,
        )

        assert result.success is False
        assert result.solution_code is None
        assert result.reward_signal == 0.0

    # ---- RECORD_DISCOVERY ----

    def test_handle_record_discovery_success(self, handler, state, session):
        session.current_best_code = "def solve(): return 42"
        session.current_best_verified = True
        session.current_best_is_novel = True
        session.problem_signature = "sig123"
        session.actions_taken = ["DECOMPOSE", "SYNTHESIZE"]
        state.correctness_confidence = 0.99
        state.current_complexity = "O(1)"
        state.current_space_complexity = "O(1)"

        expected_hash = hashlib.sha256(
            "def solve(): return 42".encode(),
        ).hexdigest()[:8]
        expected_name = f"discovered_{expected_hash}"

        params = ActionParams(action=CognitiveAction.RECORD_DISCOVERY)
        result = handler.dispatch(
            CognitiveAction.RECORD_DISCOVERY, params, state, session,
        )

        handler.kg.record_new_algorithm.assert_called_once_with(
            expected_name, "sig123", ["DECOMPOSE", "SYNTHESIZE"],
        )
        assert result.success is True
        assert result.output == {"algorithm_name": expected_name}
        assert result.solution_code == "def solve(): return 42"
        assert result.correctness == 0.99
        assert result.time_complexity == "O(1)"
        assert result.space_complexity == "O(1)"
        assert result.reward_signal == 3.0

    def test_handle_record_discovery_not_verified(self, handler, state, session):
        session.current_best_code = "code"
        session.current_best_verified = False
        session.current_best_is_novel = True

        params = ActionParams(action=CognitiveAction.RECORD_DISCOVERY)
        result = handler.dispatch(
            CognitiveAction.RECORD_DISCOVERY, params, state, session,
        )

        handler.kg.record_new_algorithm.assert_not_called()
        assert result.success is False
        assert result.error == "Nothing verified+novel to record"

    def test_handle_record_discovery_not_novel(self, handler, state, session):
        session.current_best_code = "code"
        session.current_best_verified = True
        session.current_best_is_novel = False

        params = ActionParams(action=CognitiveAction.RECORD_DISCOVERY)
        result = handler.dispatch(
            CognitiveAction.RECORD_DISCOVERY, params, state, session,
        )

        handler.kg.record_new_algorithm.assert_not_called()
        assert result.success is False

    def test_handle_record_discovery_no_code(self, handler, state, session):
        session.current_best_code = None
        session.current_best_verified = True
        session.current_best_is_novel = True

        params = ActionParams(action=CognitiveAction.RECORD_DISCOVERY)
        result = handler.dispatch(
            CognitiveAction.RECORD_DISCOVERY, params, state, session,
        )

        handler.kg.record_new_algorithm.assert_not_called()
        assert result.success is False


class TestQuickVerify:
    @pytest.fixture
    def handler(self):
        return ActionHandler(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

    def test_no_examples(self, handler):
        assert handler._quick_verify("result", None) == 0.5
        assert handler._quick_verify("result", []) == 0.5

    def test_result_is_none(self, handler):
        assert handler._quick_verify(None, [{"expected_output": 1}]) == 0.5

    def test_all_pass(self, handler):
        examples = [
            {"expected_output": 1},
            {"expected_output": 2},
            {"expected_output": 3},
        ]
        assert handler._quick_verify(1, examples) == 1 / 3

    def test_all_match_single_result(self, handler):
        examples = [{"expected_output": 42}, {"expected_output": 42}]
        assert handler._quick_verify(42, examples) == 1.0

    def test_partial_match(self, handler):
        examples = [
            {"expected_output": 1},
            {"expected_output": 2},
            {"expected_output": 3},
            {"expected_output": 4},
        ]
        assert handler._quick_verify(2, examples) == 0.25

    def test_none_match(self, handler):
        examples = [{"expected_output": 1}, {"expected_output": 2}]
        assert handler._quick_verify(99, examples) == 0.0

    def test_examples_without_expected_output(self, handler):
        examples = [{"input": 1}, {"input": 2}]
        assert handler._quick_verify("x", examples) == 0.0

    def test_dict_input_matched_on_expected_output_key(self, handler):
        examples = [{"expected_output": 42}]
        assert handler._quick_verify(42, examples) == 1.0
        assert handler._quick_verify(0, examples) == 0.0


class TestIsBetterComplexity:
    @pytest.fixture
    def handler(self):
        return ActionHandler(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

    def test_o1_better_than_on(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(1)", "O(n)") is True

    def test_on_worse_than_o1(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(n)", "O(1)") is False

    def test_equal_complexity(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(n)", "O(n)") is False

    def test_o_log_n_better_than_o_n(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(log n)", "O(n)") is True

    def test_o_n_log_n_better_than_o_n2(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(n log n)", "O(n^2)") is True

    def test_unknown_is_worse_than_everything(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "unknown", "O(n)") is False
        assert ActionHandler._is_better_complexity(ActionHandler, "unknown", "O(2^n)") is False
        assert ActionHandler._is_better_complexity(ActionHandler, "unknown", "O(1)") is False

    def test_everything_better_than_unknown(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(1)", "unknown") is True
        assert ActionHandler._is_better_complexity(ActionHandler, "O(2^n)", "unknown") is True

    def test_unrecognized_returns_false(self):
        assert ActionHandler._is_better_complexity(ActionHandler, "O(n!)", "O(n)") is False
        assert ActionHandler._is_better_complexity(ActionHandler, "O(n)", "O(n!)") is False

    def test_full_order_is_strict(self):
        order = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)", "unknown"]
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if i < j:
                    assert ActionHandler._is_better_complexity(ActionHandler, a, b) is True
                else:
                    assert ActionHandler._is_better_complexity(ActionHandler, a, b) is False
