# core/rl/__init__.py

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "AAlgoIEnv",
    "ReplayBuffer", "EpisodeBuffer", "PrioritizedReplayBuffer",
    "RewardShaper", "AdaptiveRewardShaper", "CurriculumRewardShaper",
    "PPOAgent", "AttentionActorCritic",
    "HighLevelController", "LowLevelController", "HierarchicalAgent",
    "CuriosityModule", "WorldModel", "MetaLearner", "MultiTaskAgent",
]

_POWERHOUSE_CLASSES = [
    "HighLevelController", "LowLevelController", "HierarchicalAgent",
    "CuriosityModule", "WorldModel", "MetaLearner", "MultiTaskAgent",
]

if _TORCH_AVAILABLE:
    from aalgoi.core.rl.environment import AAlgoIEnv
    from aalgoi.core.rl.replay_buffer import ReplayBuffer, EpisodeBuffer, PrioritizedReplayBuffer
    from aalgoi.core.rl.reward_shaper import RewardShaper, AdaptiveRewardShaper, CurriculumRewardShaper
    from aalgoi.core.rl.agents.selection_agent import PPOAgent, AttentionActorCritic
else:
    AAlgoIEnv = None
    ReplayBuffer = None
    EpisodeBuffer = None
    PrioritizedReplayBuffer = None
    RewardShaper = None
    AdaptiveRewardShaper = None
    CurriculumRewardShaper = None
    PPOAgent = None
    AttentionActorCritic = None


def __getattr__(name):
    if name in _POWERHOUSE_CLASSES:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RL powerhouse agents. pip install aalgoi[torch]")
        import importlib
        mod = importlib.import_module("aalgoi.core.rl.powerhouse_agent")
        return getattr(mod, name)
    raise AttributeError(f"module 'aalgoi.core.rl' has no attribute {name}")
