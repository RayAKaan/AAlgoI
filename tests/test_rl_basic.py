import sys
import numpy as np
from algorithms.primitives import PRIMITIVES
from core.problem_spec import ProblemSpec, ProblemType


def test_replay_buffer():
    from core.rl.replay_buffer import ReplayBuffer, EpisodeBuffer
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
    from core.rl.reward_shaper import RewardShaper, AdaptiveRewardShaper
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


def test_actor_critic_network():
    from core.rl.agents.selection_agent import ActorCriticNetwork
    net = ActorCriticNetwork(state_dim=200, num_actions=25)
    import torch
    state = torch.randn(1, 200)
    policy, value = net(state)
    assert policy.shape == (1, 25)
    assert value.shape == (1, 1)
    assert abs(policy.sum().item() - 1.0) < 1e-5

    action, lp, v = net.get_action(state, deterministic=True)
    assert 0 <= action < 25
    assert isinstance(lp, float)
    assert isinstance(v, float)


def test_ppo_agent():
    from core.rl.agents.selection_agent import PPOAgent
    agent = PPOAgent(state_dim=200, num_actions=25)
    state = np.random.randn(200).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    assert 0 <= action < 25

    agent.store_transition(state, action, 1.0, False, log_prob, value)
    agent.store_transition(state, action, 0.5, True, log_prob, value)
    assert len(agent.buffer) == 2


def test_environment_reset():
    from core.rl.environment import AAlgoIEnv
    env = AAlgoIEnv(algorithm_registry=PRIMITIVES, config={"state_dim": 200})
    state, info = env.reset()
    assert state.shape == (200,)
    assert "episode" in info
    assert "data_size" in info


def test_environment_step():
    from core.rl.environment import AAlgoIEnv
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
    from core.rl.environment import AAlgoIEnv
    from core.rl.agents.selection_agent import PPOAgent

    env = AAlgoIEnv(algorithm_registry=PRIMITIVES, config={
        "state_dim": 200, "batch_size": 4, "update_epochs": 2,
        "max_steps_per_episode": 3,
    })
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        config={"batch_size": 4, "update_epochs": 2},
    )

    rewards = []
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, done, log_prob, value)
            episode_reward += reward
            state = next_state
        rewards.append(episode_reward)

        if len(agent.buffer) >= agent.batch_size:
            stats = agent.train()
            assert stats is not None
            assert "policy_loss" in stats

    assert len(rewards) == 3


def test_meta_controller_rl_agent_created():
    from core.meta_controller import UniversalMetaController
    mc = UniversalMetaController(config={"rl": {"state_dim": 200}})
    assert mc.rl_agent is not None
