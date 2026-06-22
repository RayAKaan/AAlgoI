"""
Next-generation RL agent with:
- Hierarchical multi-agent architecture
- Intrinsic curiosity for exploration
- Model-based planning via world model
- Meta-learning (MAML-style fast adaptation)
- Multi-task shared representations
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ============================================
# HIERARCHICAL RL: High-level + Low-level
# ============================================

class HighLevelController(nn.Module):
    """
    Chooses strategy: greedy, optimal, hybrid, genetic, llm
    """

    def __init__(self, state_dim: int = 200, num_strategies: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_strategies),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(state), dim=-1)


class LowLevelController(nn.Module):
    """
    Given strategy, chooses specific algorithm within that strategy.
    """

    def __init__(self, state_dim: int = 200, num_algorithms: int = 100):
        super().__init__()
        self.strategy_encoder = nn.Embedding(5, 64)

        self.net = nn.Sequential(
            nn.Linear(state_dim + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_algorithms),
        )

    def forward(self, state: torch.Tensor, strategy: torch.Tensor) -> torch.Tensor:
        strategy_emb = self.strategy_encoder(strategy)
        combined = torch.cat([state, strategy_emb], dim=-1)
        return F.softmax(self.net(combined), dim=-1)


class HierarchicalAgent(nn.Module):
    """Combined high-level + low-level controller."""

    def __init__(self, state_dim: int = 200, num_strategies: int = 5, num_algorithms: int = 100):
        super().__init__()
        self.high = HighLevelController(state_dim, num_strategies)
        self.low = LowLevelController(state_dim, num_algorithms)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strategy_probs = self.high(state)
        strategy = strategy_probs.argmax(dim=-1, keepdim=True)
        algo_probs = self.low(state, strategy.squeeze(-1))
        return strategy_probs, algo_probs


# ============================================
# CURIOSITY-DRIVEN EXPLORATION
# ============================================

class CuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) - rewards exploring novel states.
    """

    def __init__(self, state_dim: int = 200, action_dim: int = 100):
        super().__init__()

        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def compute_intrinsic_reward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        action_onehot = F.one_hot(action, num_classes=100).float()
        action_onehot = action_onehot.view(action.size(0), -1)

        predicted_next = self.forward_model(torch.cat([state, action_onehot], dim=-1))
        intrinsic_reward = F.mse_loss(predicted_next, next_state, reduction="none").mean(dim=-1)

        return intrinsic_reward * 0.1


# ============================================
# WORLD MODEL (Model-Based RL)
# ============================================

class WorldModel(nn.Module):
    """
    Learns dynamics: state + action -> next_state + reward
    Enables planning without execution.
    """

    def __init__(self, state_dim: int = 200, action_dim: int = 100):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def update(self, state: torch.Tensor, action: torch.Tensor,
               actual_reward: float, actual_next_state: torch.Tensor | None = None) -> float:
        """Update world model with real transition data."""
        self.optimizer.zero_grad()

        action = action.to(dtype=torch.long)
        state = state.to(dtype=torch.float)
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        action_onehot = action_onehot.view(action.size(0), -1)
        inp = torch.cat([state, action_onehot], dim=-1)

        pred_reward = self.reward_net(inp)
        reward_target = torch.tensor([actual_reward], dtype=torch.float)
        reward_loss = F.mse_loss(pred_reward.view(-1), reward_target)

        if actual_next_state is not None:
            pred_next_state = self.transition_net(inp)
            state_loss = F.mse_loss(pred_next_state, actual_next_state)
            total_loss = reward_loss + state_loss
        else:
            total_loss = reward_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def generate_hard_state(self, state_dim: int | None = None) -> torch.Tensor:
        """Generate a state predicted to yield low reward (hard problem)."""
        dim = state_dim if state_dim is not None else self.state_dim
        state = torch.randn(1, dim, requires_grad=True)
        optimizer = torch.optim.Adam([state], lr=0.1)

        for _ in range(50):
            optimizer.zero_grad()

            action = torch.tensor([0], dtype=torch.long)
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float().view(1, -1)
            inp = torch.cat([state, action_onehot], dim=-1)
            pred_reward = self.reward_net(inp)

            loss = -pred_reward
            loss.backward()
            optimizer.step()

        return state.detach()

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = action.to(dtype=torch.long)
        state = state.to(dtype=torch.float)
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        action_onehot = action_onehot.view(action.size(0), -1)
        inp = torch.cat([state, action_onehot], dim=-1)

        next_state_pred = self.transition_net(inp)
        reward_pred = self.reward_net(inp)

        return next_state_pred, reward_pred

    def plan_ahead(self, state: torch.Tensor, num_steps: int = 5) -> int:
        """Imagine future trajectories without execution."""
        best_action = 0
        best_reward = float("-inf")

        for action_candidate in range(10):
            current = state.clone()
            total_reward = 0.0

            for _ in range(num_steps):
                action = torch.tensor([[action_candidate]])
                next_s, r = self.predict(current, action)
                total_reward += r.item()
                current = next_s

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action_candidate

        return best_action


# ============================================
# META-LEARNING (MAML-style)
# ============================================

class MetaLearner:
    """
    Learns initialization that adapts quickly to new problem types.
    """

    def __init__(self, model: nn.Module, meta_lr: float = 1e-3, inner_lr: float = 1e-2):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr

    def meta_train(self, task_batch: list[tuple[dict, dict]]) -> float:
        """
        task_batch: list of (support_set, query_set) for different problem types
        """
        meta_loss = 0.0

        for support, query in task_batch:
            adapted = self._clone_model()

            for _ in range(5):
                support_loss = self._compute_loss(adapted, support)
                grads = torch.autograd.grad(support_loss, adapted.parameters(), create_graph=True)
                adapted = self._apply_gradients(adapted, grads)

            query_loss = self._compute_loss(adapted, query)
            meta_loss += query_loss

        meta_loss /= len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def fast_adapt(self, new_problem_data: dict, num_steps: int = 5) -> None:
        """Quickly adapt to new problem type in a few gradient steps."""
        for _ in range(num_steps):
            loss = self._compute_loss(self.model, new_problem_data)
            grads = torch.autograd.grad(loss, self.model.parameters())
            for param, grad in zip(self.model.parameters(), grads):
                param.data -= self.inner_lr * grad

    def _clone_model(self) -> nn.Module:
        clone = type(self.model)(
            *[p.clone() for p in self.model.parameters()]
        )
        return clone

    def _compute_loss(self, model: nn.Module, data: dict) -> torch.Tensor:
        states = torch.FloatTensor(data.get("states", []))
        actions = torch.LongTensor(data.get("actions", []))
        if len(states) == 0:
            return torch.tensor(0.0, requires_grad=True)
        logits, _ = model(states)
        loss = F.cross_entropy(logits, actions)
        return loss

    def _apply_gradients(self, model: nn.Module, grads: list[torch.Tensor]) -> nn.Module:
        for param, grad in zip(model.parameters(), grads):
            param.data -= self.inner_lr * grad
        return model


# ============================================
# MULTI-TASK RL
# ============================================

class MultiTaskAgent(nn.Module):
    """
    Single agent that handles all problem types with shared representations.
    """

    def __init__(self, state_dim: int = 200, num_algorithms: int = 100, num_tasks: int = 10):
        super().__init__()
        self.task_encoder = nn.Embedding(num_tasks, 64)

        self.shared = nn.Sequential(
            nn.Linear(state_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, num_algorithms)
        self.value_head = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, task_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_emb = self.task_encoder(task_id)
        combined = torch.cat([state, task_emb], dim=-1)
        features = self.shared(combined)
        policy = F.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        return policy, value

    def get_action(self, state: torch.Tensor, task_id: torch.Tensor,
                   deterministic: bool = False) -> tuple[int, float, float]:
        """Sample action from policy. Returns (action, log_prob, value)."""
        policy, value = self.forward(state, task_id)

        dist = Categorical(policy)

        if deterministic:
            action = policy.argmax(dim=-1)
            log_prob = torch.zeros(1)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()
