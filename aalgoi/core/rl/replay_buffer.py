import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, info: dict = None):
        self.buffer.append({
            "state": state, "action": action, "reward": reward,
            "next_state": next_state, "done": done, "info": info or {}
        })

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "states": np.array([t["state"] for t in batch]),
            "actions": np.array([t["action"] for t in batch]),
            "rewards": np.array([t["reward"] for t in batch]),
            "next_states": np.array([t["next_state"] for t in batch]),
            "dones": np.array([t["done"] for t in batch]),
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class EpisodeBuffer:
    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.log_probs: list = []
        self.values: list = []

    def push(self, state: np.ndarray, action: int, reward: float,
             done: bool, log_prob: float = None, value: float = None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)

    def get(self) -> dict[str, np.ndarray]:
        result = {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
        }
        if self.log_probs:
            result["log_probs"] = np.array(self.log_probs)
        if self.values:
            result["values"] = np.array(self.values)
        return result

    def __len__(self) -> int:
        return len(self.states)

    def clear(self):
        for attr in ("states", "actions", "rewards", "dones", "log_probs", "values"):
            getattr(self, attr).clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done, info=None, priority=None):
        super().push(state, action, reward, next_state, done, info)
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[dict, np.ndarray, np.ndarray]:
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = {
            k: np.array([self.buffer[i][k] for i in indices])
            for k in ("state", "action", "reward", "next_state", "done")
        }
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
