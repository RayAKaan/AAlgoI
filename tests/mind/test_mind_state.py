import pytest
import torch

from aalgoi.core.mind.cognitive_actions import ActionResult, CognitiveAction
from aalgoi.core.mind.mind_state import MindState, build_data_profile


class TestBuildDataProfile:
    def test_none_input(self):
        prof = build_data_profile(None)
        assert prof.shape == (64,)
        assert torch.all(prof == 0)

    def test_integer_list_input(self):
        prof = build_data_profile([3, 1, 4, 1, 5])
        assert prof.shape == (64,)
        assert prof[2] > 0  # log size

    def test_sorted_array(self):
        prof = build_data_profile([1, 2, 3, 4, 5])
        assert prof[32] == 1.0  # sorted ascending

    def test_reverse_sorted(self):
        prof = build_data_profile([5, 4, 3, 2, 1])
        assert prof[33] == 1.0  # sorted descending

    def test_all_same(self):
        prof = build_data_profile([7, 7, 7, 7])
        assert prof[34] == 1.0  # all identical

    def test_dict_wrapped_data(self):
        prof = build_data_profile({"nums": [1, 2, 3]})
        assert prof[2] > 0

    def test_string_input(self):
        prof = build_data_profile("hello world")
        assert prof[48] > 0  # length

    def test_clamped_range(self):
        prof = build_data_profile([10**10])
        assert prof[16] <= 1.0


class TestMindState:
    @pytest.fixture
    def state(self):
        return MindState(
            problem_text="sort this array",
            problem_signature="abc123",
            data_features=torch.zeros(64),
        )

    def test_initial_values(self, state):
        assert state.step == 0
        assert state.total_reward == 0.0
        assert not state.is_terminal
        assert state.current_complexity == "unknown"
        assert state.actions_taken == []

    def test_to_tensor_returns_correct_shapes(self, state):
        tensors = state.to_tensor()
        assert tensors["data_features"].shape == (64,)
        assert tensors["action_history"].shape == (20, 25)
        assert tensors["kg_neighborhood"].shape == (128,)
        assert tensors["scalars"].shape == (8,)

    def test_update_after_action(self, state):
        result = ActionResult(
            action=CognitiveAction.IDENTIFY_STRUCTURE,
            success=True,
            output=None,
            solution_code="def solve(): pass",
            correctness=0.9,
            time_complexity="O(n log n)",
            space_complexity=None,
            reward_signal=0.5,
        )

        new_state = state.update_after_action(
            CognitiveAction.IDENTIFY_STRUCTURE, result
        )

        assert new_state.step == 1
        assert len(new_state.actions_taken) == 1
        assert new_state.total_reward == 0.5
        assert new_state.correctness_confidence == 0.9
        assert new_state.current_solution_code == "def solve(): pass"

    def test_update_keeps_original_unchanged(self, state):
        result = ActionResult(
            action=CognitiveAction.IDENTIFY_STRUCTURE,
            success=True,
            output=None,
            solution_code="code",
            correctness=1.0,
            time_complexity="O(1)",
            space_complexity=None,
            reward_signal=1.0,
        )

        new_state = state.update_after_action(
            CognitiveAction.IDENTIFY_STRUCTURE, result
        )

        assert state.step == 0
        assert new_state.step == 1
