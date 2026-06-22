from typing import Any

from aalgoi.core.mind.cognitive_actions import (
    ActionHandler,
    ActionParams,
    ActionResult,
    CognitiveAction,
)
from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph
from aalgoi.core.mind.mind_state import MindState
from aalgoi.core.mind.model_config import DEFAULT_CONFIG, MindConfig
from aalgoi.core.mind.rl_mind import AlgorithmicMind, MindOutput
from aalgoi.core.mind.safety_manager import MindSafetyManager
from aalgoi.core.mind.solving_loop import MindSolvingLoop, ThinkingSession, UniversalSolution


def create_mind(
    config: MindConfig | None = None,
    solver: Any = None,
    synthesizer: Any = None,
    persist_dir: str = "~/.aalgoi",
) -> MindSolvingLoop:
    from pathlib import Path

    from aalgoi.core.mind.adapters import (
        ExecutorAdapter,
        ProverAdapter,
        SynthesizerAdapter,
    )
    from aalgoi.core.reasoning.comprehension_engine import DeepComprehensionEngine
    from aalgoi.core.reasoning.correctness_prover import CorrectnessProver

    cfg = config or DEFAULT_CONFIG

    mind = AlgorithmicMind(cfg)

    kg_path = Path(persist_dir).expanduser() / "kg"
    kg = AlgorithmicKnowledgeGraph(kg_path)

    executor = ExecutorAdapter(solver)
    synth = SynthesizerAdapter(synthesizer)
    prover = ProverAdapter(CorrectnessProver())

    comprehension = DeepComprehensionEngine()

    handler = ActionHandler(
        kg=kg,
        executor=executor,
        synthesizer=synth,
        prover=prover,
        comprehension=comprehension,
    )

    safety = MindSafetyManager(Path(persist_dir).expanduser())

    weights_path = Path(persist_dir).expanduser() / "personal_model.pt"
    if weights_path.exists():
        import torch
        mind.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        base_path = Path(__file__).parent / "pretrained" / "base_v2.pt"
        if base_path.exists():
            import torch
            mind.load_state_dict(torch.load(base_path, map_location="cpu"))

    loop = MindSolvingLoop(
        mind=mind,
        kg=kg,
        action_handler=handler,
        safety=safety,
    )

    return loop


__all__ = [
    "AlgorithmicMind", "MindOutput",
    "CognitiveAction", "ActionParams", "ActionResult", "ActionHandler",
    "MindState",
    "MindSolvingLoop", "ThinkingSession", "UniversalSolution",
    "AlgorithmicKnowledgeGraph",
    "MindSafetyManager",
    "MindConfig", "DEFAULT_CONFIG",
    "create_mind",
]
