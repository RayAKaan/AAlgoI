import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

STATE_DIM  = 42
ALGO_DIM   = 32
HIDDEN_DIM = 128


class AttentionActorCritic(nn.Module):
    """
    Attention-based Actor-Critic network.

    Instead of a fixed-size linear output head, uses attention over
    algorithm embeddings to produce a variable-size action space.
    Adding new algorithms = adding new embedding vectors, no retraining.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        algo_dim:  int = ALGO_DIM,
        hidden_dim: int = HIDDEN_DIM,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.query_proj = nn.Linear(hidden_dim, algo_dim)
        self.key_proj   = nn.Linear(algo_dim, algo_dim)
        self.value_proj = nn.Linear(algo_dim, algo_dim)

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.scale = algo_dim ** 0.5

    def forward(
        self,
        state: torch.Tensor,
        algo_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.encoder(state)
        features = torch.nan_to_num(features, nan=0.0)

        query = self.query_proj(features)
        query = torch.nan_to_num(query, nan=0.0)

        keys   = self.key_proj(algo_embeddings)
        keys   = torch.nan_to_num(keys, nan=0.0)
        values = self.value_proj(algo_embeddings)
        values = torch.nan_to_num(values, nan=0.0)

        attn_scores = torch.matmul(
            query.unsqueeze(1),
            keys.T.unsqueeze(0)
        ).squeeze(1) / self.scale

        policy = F.softmax(attn_scores / self.temperature, dim=-1)
        value  = self.critic(features)

        return policy, value

    def get_action(
        self,
        state: torch.Tensor,
        algo_embeddings: torch.Tensor,
        deterministic: bool = False,
        candidate_mask: torch.Tensor = None,
    ) -> tuple[int, float, float]:
        policy, value = self.forward(state, algo_embeddings)

        if candidate_mask is not None:
            policy = policy * candidate_mask
            policy = policy / policy.sum(dim=-1, keepdim=True)

        dist = Categorical(policy)
        if deterministic:
            action = policy.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


class PPOAgent:
    _sd_cache: dict = {}

    @classmethod
    def _get_cached_state_dict(cls, path: str) -> dict | None:
        import os as _os
        path = _os.path.abspath(_os.path.expanduser(path))
        if path not in cls._sd_cache:
            if not _os.path.exists(path):
                return None
            try:
                cls._sd_cache[path] = torch.load(path, map_location='cpu', weights_only=False)
            except Exception as e:
                logger.warning(f"[PPOAgent] Failed to load {path}: {e}")
                return None
        return cls._sd_cache.get(path)

    @classmethod
    def _clear_cache(cls) -> None:
        cls._sd_cache.clear()

    def __init__(self, state_dim: int = STATE_DIM, config: dict = None):
        self.config = config or {}
        self.state_dim = state_dim

        hidden_dim = self.config.get("hidden_dim", HIDDEN_DIM)
        self.network = AttentionActorCritic(
            state_dim=state_dim,
            algo_dim=ALGO_DIM,
            hidden_dim=hidden_dim,
        )

        lr = self.config.get("learning_rate", 3e-4)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.clip_epsilon = self.config.get("clip_epsilon", 0.2)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.update_epochs = self.config.get("update_epochs", 10)
        self.batch_size = self.config.get("batch_size", 64)

        from aalgoi.core.rl.replay_buffer import EpisodeBuffer
        self.replay_buffer = EpisodeBuffer()
        self.buffer = self.replay_buffer

        self.training_stats = {
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy": deque(maxlen=100),
            "total_loss": deque(maxlen=100),
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self._algo_embeddings: torch.Tensor | None = None
        self._algo_names: list[str] = []

        self.lora_adapter = None

        self.network.eval()
        self._calibrated_temperature = self.config.get("calibration_temperature", 1.5)
        logger.info(f"PPOAgent initialized: state_dim={state_dim}, device={self.device}")

    def get_calibrated_probs(
        self,
        state: np.ndarray,
        temperature: float | None = None,
    ) -> tuple[np.ndarray, float]:
        temp = temperature if temperature is not None else self._calibrated_temperature
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        algo_t = self._algo_embeddings
        with torch.no_grad():
            features = self.network.encoder(state_t)
            features = torch.nan_to_num(features, nan=0.0)
            query = self.network.query_proj(features)
            keys = self.network.key_proj(algo_t)
            attn = torch.matmul(query.unsqueeze(1), keys.T.unsqueeze(0)).squeeze(1)
            attn = attn / self.network.scale
            probs = F.softmax(attn / temp, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return probs.squeeze(0).cpu().numpy(), entropy.item()

    def update_algo_embeddings(
        self,
        embeddings: torch.Tensor,
        algo_names: list[str],
    ) -> None:
        self._algo_embeddings = embeddings.to(self.device)
        self._algo_names = algo_names

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        candidate_mask: list = None,
    ) -> tuple[int, float, float]:
        if self._algo_embeddings is None:
            raise RuntimeError(
                "No algorithm embeddings loaded. "
                "Call update_algo_embeddings() first."
            )

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        mask_t = None
        if candidate_mask is not None:
            n_algos = self._algo_embeddings.size(0)
            mask_t = torch.zeros(n_algos, device=self.device)
            mask_t[candidate_mask] = 1.0

        was_training = self.network.training
        if deterministic:
            self.network.eval()

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(
                state_t, self._algo_embeddings,
                deterministic=deterministic,
                candidate_mask=mask_t,
            )

        if deterministic:
            self.network.train(was_training)

        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.replay_buffer.push(state, action, reward, done, log_prob, value)

    def train(self) -> dict | None:
        if len(self.replay_buffer) < self.batch_size:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        if self._algo_embeddings is None:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        batch = self.replay_buffer.get()
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        old_log = torch.FloatTensor(batch["log_probs"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        old_vals = torch.FloatTensor(batch["values"]).to(self.device)

        advantages, returns = self._compute_gae(rewards, old_vals, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        algo_embeds = self._algo_embeddings

        for _ in range(self.update_epochs):
            policy, values = self.network(states, algo_embeds)
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
        self.replay_buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(self.device)
        return advantages, advantages + values

    def save(self, path: str) -> None:
        import os as _os
        sd = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": dict(self.training_stats),
            "algo_names": self._algo_names,
        }
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        torch.save(sd, path)
        self.__class__._sd_cache[_os.path.abspath(path)] = sd
        logger.info(f"Model saved to {path}")

    def load(self, path: str, weights_only: bool = False) -> bool:
        sd = self._get_cached_state_dict(path)
        if sd is None:
            logger.warning(f"Checkpoint not found: {path}")
            return False
        try:
            ckpt_sd = sd["network_state"]

            # Handle LoRA key mapping: if checkpoint has base keys (from LoRALinear)
            # but current network expects flat keys (nn.Linear), or vice versa
            model_keys = set(self.network.state_dict().keys())
            has_base_keys = any('.base.' in k for k in ckpt_sd)
            model_has_base = any('.base.' in k for k in model_keys)

            if has_base_keys and not model_has_base:
                # Checkpoint has LoRA keys, network expects flat keys
                mapped = {}
                for k, v in ckpt_sd.items():
                    if k.endswith('.base.weight'):
                        mapped[k.replace('.base.weight', '.weight')] = v
                    elif k.endswith('.base.bias'):
                        mapped[k.replace('.base.bias', '.bias')] = v
                    elif '.lora_' not in k:
                        mapped[k] = v
                self.network.load_state_dict(mapped, strict=False)
            elif not has_base_keys and model_has_base:
                # Checkpoint has flat keys, network has LoRALinear
                mapped = {}
                for k, v in ckpt_sd.items():
                    if k.endswith('.weight') and k.replace('.weight', '.base.weight') in model_keys:
                        mapped[k.replace('.weight', '.base.weight')] = v
                    elif k.endswith('.bias') and k.replace('.bias', '.base.bias') in model_keys:
                        mapped[k.replace('.bias', '.base.bias')] = v
                    else:
                        mapped[k] = v
                self.network.load_state_dict(mapped, strict=False)
            else:
                self.network.load_state_dict(ckpt_sd)

            if "optimizer_state" in sd:
                try:
                    self.optimizer.load_state_dict(sd["optimizer_state"])
                except Exception:
                    pass
            if "algo_names" in sd:
                self._algo_names = sd["algo_names"]
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            return False

    def reload_adapter(self) -> None:
        if self.lora_adapter is None:
            return
        from aalgoi.core.checkpoint_manager import CheckpointManager
        manager = CheckpointManager()
        path = manager.get_current_adapter_path()
        if path:
            self.lora_adapter.load(path)

    def get_stats(self) -> dict:
        return {
            k: {"mean": float(np.mean(v)) if v else 0.0, "std": float(np.std(v)) if v else 0.0}
            for k, v in self.training_stats.items()
        }
