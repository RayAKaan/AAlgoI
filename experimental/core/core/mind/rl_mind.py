from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from aalgoi.core.mind.cognitive_actions import ActionParams, CognitiveAction
from aalgoi.core.mind.mind_state import MindState
from aalgoi.core.mind.model_config import DEFAULT_CONFIG, MindConfig

if TYPE_CHECKING:
    from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph

if TYPE_CHECKING:
    pass


@dataclass
class MindOutput:
    action: CognitiveAction
    action_params: ActionParams
    action_probs: torch.Tensor
    value: float
    confidence: float


class AlgorithmicMind(nn.Module):
    def __init__(self, config: MindConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_len, config.hidden_dim
        )
        self.text_dropout = nn.Dropout(config.dropout)

        state_feature_dim = (
            config.structural_feature_dim
            + 8
            + 128
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        self.action_history_gru = nn.GRU(
            input_size=config.n_cognitive_actions,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        self.algorithm_embeddings = nn.Embedding(
            config.n_algorithms,
            config.algo_emb_dim,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.n_cognitive_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        self.param_heads = nn.ModuleDict({
            "select_algorithm": nn.Linear(config.hidden_dim, config.n_algorithms),
            "modify_algorithm": nn.Linear(config.hidden_dim, 16),
            "apply_optimization": nn.Linear(config.hidden_dim, 8),
            "synthesize_new": nn.Linear(config.hidden_dim, config.n_principles),
        })

        self.log_temperature = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        problem_tokens: torch.Tensor,
        state_features: torch.Tensor,
        action_history: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        problem_tokens.shape[0]
        device = problem_tokens.device

        positions = torch.arange(problem_tokens.shape[1], device=device)
        text_emb = (
            self.token_embedding(problem_tokens)
            + self.position_embedding(positions).unsqueeze(0)
        )
        text_emb = self.text_dropout(text_emb)

        state_emb = self.state_encoder(state_features)
        state_token = state_emb.unsqueeze(1)

        _, history_hidden = self.action_history_gru(action_history)
        history_emb = history_hidden[-1]
        history_token = history_emb.unsqueeze(1)

        all_tokens = torch.cat([text_emb, state_token, history_token], dim=1)

        attended = self.transformer(all_tokens)

        fused = attended.mean(dim=1)

        logits = self.policy_head(fused)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        temperature = self.log_temperature.exp().clamp(0.5, 5.0)
        action_probs = F.softmax(logits / temperature, dim=-1)

        value = self.value_head(fused)

        return action_probs, value, fused

    def get_action_params(
        self,
        action: CognitiveAction,
        fused_state: torch.Tensor,
        kg: "AlgorithmicKnowledgeGraph | None" = None,
    ) -> ActionParams:
        params = ActionParams(action=action)

        if action == CognitiveAction.SELECT_ALGORITHM:
            algo_logits = self.param_heads["select_algorithm"](fused_state)
            algo_idx = algo_logits.argmax(dim=-1).item()
            if kg is not None:
                params.algorithm_id = kg.index_to_algorithm_name(algo_idx)

        elif action == CognitiveAction.MODIFY_ALGORITHM:
            mod_logits = self.param_heads["modify_algorithm"](fused_state)
            mod_types = [
                "add_memoization", "convert_to_iterative",
                "add_early_termination", "optimize_inner_loop",
                "add_pruning", "restructure_recursion",
                "convert_to_dp", "add_binary_search",
                "apply_two_pointer", "apply_sliding_window",
                "reduce_space", "vectorize",
                "add_hashing", "sort_first",
                "add_monotonic_stack", "apply_greedy",
            ]
            mod_idx = mod_logits.argmax(dim=-1).item()
            params.modification_type = mod_types[mod_idx % len(mod_types)]

        elif action == CognitiveAction.APPLY_OPTIMIZATION:
            opt_logits = self.param_heads["apply_optimization"](fused_state)
            opt_types = [
                "memoize", "two_pointer", "binary_search",
                "rolling_array", "monotonic_stack", "prefix_sum",
                "bit_manipulation", "greedy_exchange",
            ]
            opt_idx = opt_logits.argmax(dim=-1).item()
            params.optimization_type = opt_types[opt_idx % len(opt_types)]

        elif action == CognitiveAction.SYNTHESIZE_NEW:
            princ_logits = self.param_heads["synthesize_new"](fused_state)
            principle_names = [
                "optimal_substructure", "greedy_exchange",
                "monotonic_feasibility", "amortized_invariant",
                "divide_conquer", "hashing_fingerprint",
                "graph_flow_cut", "information_theoretic",
            ]
            princ_idx = princ_logits.argmax(dim=-1).item()
            params.principle_name = principle_names[princ_idx % len(principle_names)]

        return params

    def select_action(
        self,
        state: MindState,
        available_actions: list[CognitiveAction] | None = None,
        kg: "AlgorithmicKnowledgeGraph | None" = None,
    ) -> tuple[CognitiveAction, ActionParams, float]:
        self.eval()
        with torch.no_grad():
            tensors = state.to_tensor()
            tokens = self._tokenize_problem(state.problem_text)

            mask = torch.ones(self.config.n_cognitive_actions, dtype=torch.bool)
            if available_actions is not None:
                mask = torch.zeros(self.config.n_cognitive_actions, dtype=torch.bool)
                for a in available_actions:
                    mask[int(a)] = True

            probs, value, fused = self.forward(
                problem_tokens=tokens.unsqueeze(0),
                state_features=torch.cat([
                    tensors["data_features"],
                    tensors["scalars"],
                    tensors["kg_neighborhood"],
                ]).unsqueeze(0),
                action_history=tensors["action_history"].unsqueeze(0),
                action_mask=mask.unsqueeze(0),
            )

            dist = torch.distributions.Categorical(probs.squeeze(0))
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            action = CognitiveAction(action_idx.item())

            params = self.get_action_params(action, fused.squeeze(0), kg)

        return action, params, log_prob.item()

    def get_calibrated_probs(
        self,
        state: MindState,
    ) -> tuple[torch.Tensor, float]:
        self.eval()
        with torch.no_grad():
            tensors = state.to_tensor()
            tokens = self._tokenize_problem(state.problem_text)
            mask = torch.ones(self.config.n_cognitive_actions, dtype=torch.bool)

            probs, value, _ = self.forward(
                problem_tokens=tokens.unsqueeze(0),
                state_features=torch.cat([
                    tensors["data_features"],
                    tensors["scalars"],
                    tensors["kg_neighborhood"],
                ]).unsqueeze(0),
                action_history=tensors["action_history"].unsqueeze(0),
                action_mask=mask.unsqueeze(0),
            )

            confidence = probs.max().item()
            return probs.squeeze(0), confidence

    def store_transition(
        self,
        state: MindState,
        action: CognitiveAction,
        reward: float,
        next_state: MindState,
        done: bool,
        log_prob: float,
    ) -> None:
        if not hasattr(self, "_trajectory"):
            self._trajectory = []
        self._trajectory.append({
            "state": state,
            "action": int(action),
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "log_prob": log_prob,
        })

    def train_on_trajectory(self, optimizer: torch.optim.Optimizer) -> dict:
        if not hasattr(self, "_trajectory") or not self._trajectory:
            return {}

        self.train()
        metrics = self._ppo_update(self._trajectory, optimizer)
        self._trajectory = []
        return metrics

    def _ppo_update(
        self,
        trajectory: list[dict],
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        cfg = self.config

        returns = []
        advantages = []
        gamma = 0.99
        lam = 0.95

        values_list = [t.get("value", 0.0) for t in trajectory]
        rewards = [t["reward"] for t in trajectory]

        gae = 0.0
        for t in reversed(range(len(trajectory))):
            next_val = values_list[t + 1] if t + 1 < len(values_list) else 0.0
            delta = rewards[t] + gamma * next_val - values_list[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_list[t])

        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        total_loss = 0.0
        for epoch in range(cfg.ppo_epochs):
            for t, trans in enumerate(trajectory):
                tensors = trans["state"].to_tensor()
                tokens = self._tokenize_problem(trans["state"].problem_text)
                mask = torch.ones(cfg.n_cognitive_actions, dtype=torch.bool)

                probs, value, _ = self.forward(
                    tokens.unsqueeze(0),
                    torch.cat([
                        tensors["data_features"],
                        tensors["scalars"],
                        tensors["kg_neighborhood"],
                    ]).unsqueeze(0),
                    tensors["action_history"].unsqueeze(0),
                    mask.unsqueeze(0),
                )

                dist = torch.distributions.Categorical(probs.squeeze(0))
                action_t = torch.tensor(trans["action"])
                new_log_prob = dist.log_prob(action_t)
                old_log_prob = torch.tensor(trans["log_prob"])

                ratio = (new_log_prob - old_log_prob).exp()
                adv = advantages_t[t]
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = F.mse_loss(value.squeeze(), returns_t[t])

                entropy = dist.entropy()

                loss = (
                    policy_loss
                    + cfg.value_coeff * value_loss
                    - cfg.entropy_coeff * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()

        return {
            "ppo_loss": total_loss / (len(trajectory) * cfg.ppo_epochs),
            "trajectory_len": len(trajectory),
        }

    def _tokenize_problem(self, text: str) -> torch.Tensor:
        chars = [ord(c) % self.config.vocab_size for c in text[:self.config.max_seq_len]]
        chars += [0] * (self.config.max_seq_len - len(chars))
        return torch.tensor(chars, dtype=torch.long)

    @property
    def size_mb(self) -> float:
        params = sum(p.numel() for p in self.parameters())
        dtype_bytes = 2 if self.config.use_fp16 else 4
        return params * dtype_bytes / (1024**2)
