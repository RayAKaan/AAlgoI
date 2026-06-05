import sys
import numpy as np
from aalgoi.algorithms.primitives import PRIMITIVES
from aalgoi.core.problem_spec import ProblemSpec, ProblemType


def test_replay_buffer():
    from aalgoi.core.rl.replay_buffer import ReplayBuffer, EpisodeBuffer
    buf = ReplayBuffer(capacity=100)
    buf.push(np.zeros(10), 0, 1.0, np.ones(10), False)
    buf.push(np.zeros(10), 1, 0.5, np.ones(10), True)
    assert len(buf) == 2
    batch = buf.sample(2)
    assert "states" in batch
    assert batch["actions"].shape == (2,)
    assert batch["rewards"].shape == (2,)

    ep = EpisodeBuffer()
    ep.push(np.zeros(10), 0, 1.0, False, log_prob=-0.5, value=0.1)
    ep.push(np.ones(10), 1, 0.0, True, log_prob=-0.3, value=0.2)
    assert len(ep) == 2
    data = ep.get()
    assert "log_probs" in data
    assert data["log_probs"].shape == (2,)
    ep.clear()
    assert len(ep) == 0


def test_reward_shaper():
    from aalgoi.core.rl.reward_shaper import RewardShaper, AdaptiveRewardShaper
    shaper = RewardShaper()

    reward = shaper.compute_reward(
        is_valid=True,
        metrics={"quality_score": 0.95, "wall_time_ms": 50, "memory_mb": 10, "algorithms": []},
        context={"time_budget_ms": 500, "memory_budget_mb": 100},
    )
    assert reward > 0, f"Expected positive reward, got {reward}"

    reward = shaper.compute_reward(is_valid=False, metrics={}, context={})
    assert reward < 0, f"Expected negative reward for failure, got {reward}"

    adaptive = AdaptiveRewardShaper()
    reward_speed = adaptive.compute_reward(
        is_valid=True,
        metrics={"quality_score": 0.8, "wall_time_ms": 10, "memory_mb": 50, "algorithms": []},
        context={"time_budget_ms": 500, "memory_budget_mb": 1024, "priority": "speed"},
    )
    assert reward_speed > 0


def test_attention_actor_critic():
    from aalgoi.core.rl.agents.selection_agent import AttentionActorCritic
    import torch
    net = AttentionActorCritic(state_dim=38, algo_dim=32, hidden_dim=128)
    state = torch.randn(1, 38)
    algo_embeds = torch.randn(10, 32)
    policy, value = net(state, algo_embeds)
    assert policy.shape == (1, 10)
    assert value.shape == (1, 1)
    assert abs(policy.sum().item() - 1.0) < 1e-5

    action, lp, v = net.get_action(state, algo_embeds, deterministic=True)
    assert 0 <= action < 10
    assert isinstance(lp, float)
    assert isinstance(v, float)


def test_ppo_agent():
    from aalgoi.core.rl.agents.selection_agent import PPOAgent
    import torch
    agent = PPOAgent(state_dim=38)
    embeddings = torch.randn(5, 32)
    agent.update_algo_embeddings(embeddings, ['a', 'b', 'c', 'd', 'e'])
    state = np.random.randn(38).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    assert 0 <= action < 5

    agent.store_transition(state, action, 1.0, False, log_prob, value)
    agent.store_transition(state, action, 0.5, True, log_prob, value)
    assert len(agent.buffer) == 2


def test_environment_reset():
    from aalgoi.core.rl.environment import AAlgoIEnv
    env = AAlgoIEnv(algorithm_registry=PRIMITIVES, config={"state_dim": 200})
    state, info = env.reset()
    assert state.shape == (200,)
    assert "episode" in info
    assert "data_size" in info


def test_environment_step():
    from aalgoi.core.rl.environment import AAlgoIEnv
    env = AAlgoIEnv(algorithm_registry=PRIMITIVES, config={"state_dim": 200})
    state, _ = env.reset()
    action = 0
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (200,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "algorithm" in info
    assert "valid" in info


def test_full_training_loop():
    from aalgoi.core.rl.agents.selection_agent import PPOAgent
    import torch
    import numpy as np

    agent = PPOAgent(state_dim=38, config={"batch_size": 4, "update_epochs": 2})
    algo_names = ['a', 'b', 'c', 'd', 'e']
    embeddings = torch.randn(5, 32)
    agent.update_algo_embeddings(embeddings, algo_names)

    for episode in range(3):
        state = np.random.randn(38).astype(np.float32)
        episode_reward = 0.0
        for step in range(4):
            action, log_prob, value = agent.select_action(state)
            reward = 1.0 if action < 3 else -1.0
            done = step == 3
            agent.store_transition(state, action, reward, done, log_prob, value)
            episode_reward += reward
            state = np.random.randn(38).astype(np.float32)

        if len(agent.buffer) >= agent.batch_size:
            stats = agent.train()
            assert stats is not None
            assert "policy_loss" in stats


def test_meta_controller_rl_agent_created():
    from aalgoi.core.meta_controller import UniversalMetaController
    mc = UniversalMetaController(config={"rl": {"state_dim": 200}})
    assert mc.rl_agent is not None
