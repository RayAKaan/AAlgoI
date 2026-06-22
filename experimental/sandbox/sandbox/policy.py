"""
aalgoi.sandbox.policy — A2C policy network for algorithm selection.

Uses Advantage Actor-Critic (A2C):
  - Actor:  state → action probabilities
  - Critic: state → scalar value baseline
  - Loss:   actor_loss + 0.5*critic_loss - 0.01*entropy

A2C is chosen over REINFORCE (high variance, no baseline) and PPO
(complex, already used in the production pipeline). A2C gives reliable
convergence with minimal complexity — right for a user-facing sandbox.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch required: pip install torch")


class A2CNetwork(nn.Module):
    """
    Shared trunk → actor head + critic head.

    Architecture:
        Input:  state vector
        Trunk:  [Linear → ReLU → LayerNorm] × num_layers
        Actor:  Linear → N_algos logits
        Critic: Linear → 1 scalar value
    """

    def __init__(
        self,
        input_size: int,
        num_algorithms: int,
        hidden_size: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_algorithms = num_algorithms

        layers = []
        in_size = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.LayerNorm(hidden_size)]
            in_size = hidden_size

        self.trunk  = nn.Sequential(*layers)
        self.actor  = nn.Linear(hidden_size, num_algorithms)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class PolicyAgent:
    """
    A2C agent that learns which algorithm to select for a given problem.

    Key differences from the production PPOAgent:
    - No clipping (simpler loss)
    - No GAE (single-step advantage)
    - Entropy bonus for sustained exploration
    - Designed for interactive, user-driven training
    """

    ENTROPY_COEF  = 0.01
    CRITIC_COEF   = 0.5
    GRAD_CLIP     = 1.0
    BUFFER_LIMIT  = 512

    def __init__(
        self,
        input_size: int,
        num_algorithms: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.003,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.num_algorithms = num_algorithms
        self.gamma   = gamma
        self.epsilon = epsilon
        self._epsilon_start = epsilon
        self._epsilon_floor = 0.01
        self._epsilon_decay = 0.997

        self.net = A2CNetwork(input_size, num_algorithms, hidden_size, num_layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Experience buffer: (state, action, reward)
        self._buffer: List[Tuple[np.ndarray, int, float]] = []

        self.episodes      = 0
        self.total_reward  = 0.0
        self._reward_hist: List[float] = []

    # ── Selection ─────────────────────────────────────────────────────────

    def select(self, state: np.ndarray) -> Tuple[int, float]:
        """Epsilon-greedy selection. Returns (algo_index, confidence)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_algorithms), 0.0

        t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.net.action_probs(t)
        idx  = torch.argmax(probs, dim=-1).item()
        conf = probs[0, idx].item()
        return idx, conf

    # ── Experience recording ───────────────────────────────────────────────

    def record(self, state: np.ndarray, action: int, reward: float):
        self._buffer.append((state.copy(), action, reward))
        if len(self._buffer) > self.BUFFER_LIMIT:
            self._buffer = self._buffer[-self.BUFFER_LIMIT:]

        self.total_reward += reward
        self.episodes     += 1
        self._reward_hist.append(reward)

    # ── A2C update ────────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Single A2C gradient step over the most recent batch.
        Returns total loss, or None if buffer too small.
        """
        if len(self._buffer) < 8:
            return None

        batch      = self._buffer[-min(128, len(self._buffer)):]
        states, actions, rewards = zip(*batch)

        states_t  = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(self._discounted_returns(rewards))

        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        logits, values = self.net(states_t)
        log_probs      = F.log_softmax(logits, dim=-1)
        probs          = F.softmax(logits, dim=-1)

        advantage      = (returns_t - values.detach())

        sel_log_probs  = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        actor_loss     = -(sel_log_probs * advantage).mean()

        critic_loss    = F.mse_loss(values, returns_t)

        entropy        = -(probs * log_probs).sum(dim=-1).mean()

        loss = actor_loss + self.CRITIC_COEF * critic_loss - self.ENTROPY_COEF * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.GRAD_CLIP)
        self.opt.step()

        self.epsilon = max(self._epsilon_floor, self.epsilon * self._epsilon_decay)

        return loss.item()

    def _discounted_returns(self, rewards: tuple) -> List[float]:
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "net_state":  self.net.state_dict(),
            "opt_state":  self.opt.state_dict(),
            "epsilon":    self.epsilon,
            "episodes":   self.episodes,
            "total_reward": self.total_reward,
            "reward_hist": self._reward_hist[-2000:],
            "config": {
                "input_size":     self.net.trunk[0].in_features,
                "num_algorithms": self.num_algorithms,
                "hidden_size":    self.net.hidden_size,
                "num_layers":     sum(1 for m in self.net.trunk if isinstance(m, nn.Linear)),
                "gamma":          self.gamma,
            },
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=False)
        self.net.load_state_dict(ckpt["net_state"])
        self.opt.load_state_dict(ckpt["opt_state"])
        self.epsilon     = ckpt["epsilon"]
        self.episodes    = ckpt["episodes"]
        self.total_reward = ckpt["total_reward"]
        self._reward_hist = ckpt.get("reward_hist", [])

    @classmethod
    def from_file(cls, path: str, lr: float = 0.003) -> "PolicyAgent":
        ckpt   = torch.load(path, weights_only=False)
        cfg    = ckpt["config"]
        agent  = cls(
            input_size=cfg["input_size"],
            num_algorithms=cfg["num_algorithms"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            learning_rate=lr,
            gamma=cfg["gamma"],
        )
        agent.net.load_state_dict(ckpt["net_state"])
        agent.opt.load_state_dict(ckpt["opt_state"])
        agent.epsilon      = ckpt["epsilon"]
        agent.episodes     = ckpt["episodes"]
        agent.total_reward = ckpt["total_reward"]
        agent._reward_hist = ckpt.get("reward_hist", [])
        return agent

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        recent = float(np.mean(self._reward_hist[-20:])) if self._reward_hist else 0.0
        return {
            "episodes":      self.episodes,
            "total_reward":  round(self.total_reward, 2),
            "avg_reward":    round(self.total_reward / max(self.episodes, 1), 3),
            "recent_avg":    round(recent, 3),
            "epsilon":       round(self.epsilon, 4),
            "buffer_size":   len(self._buffer),
        }

    def reset_exploration(self):
        """Reset epsilon back to starting value (useful after fine-tuning)."""
        self.epsilon = self._epsilon_start

    def set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr
