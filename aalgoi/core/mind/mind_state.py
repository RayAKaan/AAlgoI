from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from aalgoi.core.mind.cognitive_actions import ActionResult, CognitiveAction
    from aalgoi.core.reasoning.essence import ProblemEssence


def build_data_profile(data: Any, dim: int = 64) -> torch.Tensor:
    features = np.zeros(dim, dtype=np.float32)

    if data is None:
        return torch.from_numpy(features)

    if isinstance(data, dict):
        for key in ("nums", "arr", "data", "values", "numbers"):
            if key in data:
                data = data[key]
                break
        else:
            for v in data.values():
                if isinstance(v, (list, np.ndarray)):
                    data = v
                    break

    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data) if not isinstance(data, np.ndarray) else data
        features[0] = min(arr.ndim, 8)
        features[1] = min(len(arr), 1e6) / 1e6
        features[2] = np.log10(max(len(arr), 1))
        if arr.ndim >= 2:
            features[3] = min(arr.shape[1], 1e3) / 1e3
            features[4] = np.log10(max(arr.shape[1], 1))

    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data) if not isinstance(data, np.ndarray) else data
        flat = arr.flatten()
        features[8] = float(np.issubdtype(arr.dtype, np.integer))
        features[9] = float(np.issubdtype(arr.dtype, np.floating))
        features[10] = float(arr.dtype == object)
        features[11] = float(arr.ndim >= 2)
        features[12] = float(len(flat) > 0 and isinstance(flat[0], (list, tuple)))

    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data) if not isinstance(data, np.ndarray) else data
        try:
            flat = arr.flatten().astype(np.float64)
            if len(flat) > 0:
                features[16] = np.mean(flat)
                features[17] = np.std(flat) if len(flat) > 1 else 0.0
                features[18] = np.min(flat)
                features[19] = np.max(flat)
                features[20] = np.median(flat)
                features[21] = features[19] - features[18]
                scale = max(abs(features[16]), abs(features[17]), 1.0)
                features[16:22] /= scale
        except (ValueError, TypeError):
            pass

    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data) if not isinstance(data, np.ndarray) else data
        try:
            flat = arr.flatten()
            if len(flat) > 0:
                features[24] = len(np.unique(flat)) / max(len(flat), 1)
                features[25] = float(any(flat != flat[0]))
                features[26] = float(any(flat < 0)) if np.issubdtype(arr.dtype, np.number) else 0.0
                features[27] = float(any(flat == 0)) if np.issubdtype(arr.dtype, np.number) else 0.0
        except (ValueError, TypeError):
            pass

    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data) if not isinstance(data, np.ndarray) else data
        try:
            flat = arr.flatten().astype(np.float64)
            if len(flat) > 1:
                features[32] = float(np.all(flat[:-1] <= flat[1:]))
                features[33] = float(np.all(flat[:-1] >= flat[1:]))
                features[34] = float(len(np.unique(flat)) == 1)
        except (ValueError, TypeError):
            pass

    if isinstance(data, dict):
        if "edges" in data:
            edges = data["edges"]
            n_nodes = data.get("n", data.get("num_nodes", 0))
            n_edges = len(edges)
            features[40] = min(n_nodes, 1e4) / 1e4
            features[41] = min(n_edges, 1e5) / 1e5
            if n_nodes > 0:
                features[42] = n_edges / (n_nodes * (n_nodes - 1))
                features[43] = 2 * n_edges / n_nodes

    if isinstance(data, str):
        features[48] = min(len(data), 1e4) / 1e4
        words = data.split()
        features[49] = min(len(words), 1e3) / 1e3
        features[50] = len(set(words)) / max(len(words), 1) if words else 0.0
        features[51] = np.mean([len(w) for w in words]) if words else 0.0

    features = np.clip(features, -1.0, 1.0)
    return torch.from_numpy(features)


@dataclass
class MindState:
    problem_text: str
    problem_signature: str
    data_features: torch.Tensor

    problem_essence: "ProblemEssence | None" = None
    identified_principle: str | None = None
    identified_structure: str | None = None
    target_complexity: str | None = None
    constraint_profile: dict = field(default_factory=dict)

    current_solution_code: str | None = None
    current_algorithm_name: str | None = None
    current_complexity: str = "unknown"
    current_space_complexity: str = "unknown"
    correctness_confidence: float = 0.0
    solution_verified: bool = False
    solution_is_novel: bool = False

    kg_neighborhood: torch.Tensor = field(
        default_factory=lambda: torch.zeros(128)
    )
    similar_problems_found: int = 0
    kg_candidate_algorithms: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)

    actions_taken: list["CognitiveAction"] = field(default_factory=list)
    action_results_summary: list[float] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)
    approaches_tried: int = 0

    step: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False

    def to_tensor(self, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:

        history_len = 20
        n_actions = 25
        history = torch.zeros(history_len, n_actions)
        for i, action in enumerate(self.actions_taken[-history_len:]):
            history[i, int(action)] = 1.0

        scalars = torch.tensor([
            self.correctness_confidence,
            self.approaches_tried / 50.0,
            self.step / 50.0,
            self.similar_problems_found / 10.0,
            float(self.solution_verified),
            float(self.solution_is_novel),
            float(self.problem_essence is not None),
            float(self.identified_principle is not None),
        ], dtype=torch.float32)

        return {
            "data_features": self.data_features.to(device),
            "action_history": history.to(device),
            "kg_neighborhood": self.kg_neighborhood.to(device),
            "scalars": scalars.to(device),
        }

    def update_after_action(
        self,
        action: "CognitiveAction",
        result: "ActionResult",
    ) -> "MindState":
        new_state = MindState(**self.__dict__)
        new_state.step += 1
        new_state.actions_taken = self.actions_taken + [action]
        new_state.action_results_summary = (
            self.action_results_summary + [result.reward_signal]
        )
        new_state.total_reward = self.total_reward + result.reward_signal

        if result.output and hasattr(result.output, "hidden_structure"):
            new_state.problem_essence = result.output
            new_state.identified_structure = result.output.hidden_structure
            new_state.target_complexity = result.output.time_budget

        if result.solution_code:
            if result.correctness > new_state.correctness_confidence:
                new_state.current_solution_code = result.solution_code
                new_state.correctness_confidence = result.correctness
                new_state.solution_is_novel = result.is_novel_algorithm

        if result.time_complexity:
            new_state.current_complexity = result.time_complexity
        if result.space_complexity:
            new_state.current_space_complexity = result.space_complexity

        if not result.success and new_state.current_algorithm_name:
            if new_state.current_algorithm_name not in new_state.failed_approaches:
                new_state.failed_approaches = (
                    new_state.failed_approaches + [new_state.current_algorithm_name]
                )

        return new_state
