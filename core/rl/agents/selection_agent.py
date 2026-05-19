import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int = 200, num_actions: int = 25, hidden_dim: int = 256):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        logits = self.actor(features)
        policy = torch.softmax(logits, dim=-1)
        value = self.critic(features)
        return policy, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        policy, value = self.forward(state)
        dist = Categorical(policy)
        if deterministic:
            action = policy.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


class PPOAgent:
    def __init__(self, state_dim: int = 200, num_actions: int = 25, config: Dict = None):
        self.config = config or {}
        self.state_dim = state_dim
        self.num_actions = num_actions

        hidden_dim = self.config.get("hidden_dim", 256)
        self.network = ActorCriticNetwork(state_dim, num_actions, hidden_dim)

        lr = self.config.get("learning_rate", 3e-4)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.clip_epsilon = self.config.get("clip_epsilon", 0.2)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.update_epochs = self.config.get("update_epochs", 10)
        self.batch_size = self.config.get("batch_size", 64)

        from core.rl.replay_buffer import EpisodeBuffer
        self.buffer = EpisodeBuffer()

        self.training_stats = {
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy": deque(maxlen=100),
            "total_loss": deque(maxlen=100),
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        logger.info(f"PPOAgent initialized: state_dim={state_dim}, actions={num_actions}, device={self.device}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            was_training = self.network.training
            if deterministic:
                self.network.eval()
            action, log_prob, value = self.network.get_action(tensor, deterministic=deterministic)
            if deterministic:
                self.network.train(was_training)
        return action, log_prob, value

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         done: bool, log_prob: float, value: float):
        self.buffer.push(state, action, reward, done, log_prob, value)

    def train(self) -> Optional[Dict]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.get()
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        old_log = torch.FloatTensor(batch["log_probs"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        old_vals = torch.FloatTensor(batch["values"]).to(self.device)

        advantages, returns = self._compute_gae(rewards, old_vals, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            policy, values = self.network(states)
            values = values.squeeze()
            dist = Categorical(policy)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values, returns)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.training_stats["policy_loss"].append(policy_loss.item())
        self.training_stats["value_loss"].append(value_loss.item())
        self.training_stats["entropy"].append(entropy.item())
        self.training_stats["total_loss"].append(loss.item())
        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(self.device)
        return advantages, advantages + values

    def save(self, path: str):
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": dict(self.training_stats),
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, weights_only: bool = False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(f"Model loaded from {path}")

    def get_stats(self) -> Dict:
        return {
            k: {"mean": float(np.mean(v)) if v else 0.0, "std": float(np.std(v)) if v else 0.0}
            for k, v in self.training_stats.items()
        }
