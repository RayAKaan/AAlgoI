from core.rl.environment import AAlgoIEnv
from core.rl.replay_buffer import ReplayBuffer, EpisodeBuffer, PrioritizedReplayBuffer
from core.rl.reward_shaper import RewardShaper, AdaptiveRewardShaper, CurriculumRewardShaper
from core.rl.agents.selection_agent import PPOAgent, AttentionActorCritic

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


def __getattr__(name):
    if name in _POWERHOUSE_CLASSES:
        import importlib
        mod = importlib.import_module("core.rl.powerhouse_agent")
        return getattr(mod, name)
    raise AttributeError(f"module 'core.rl' has no attribute {name}")
