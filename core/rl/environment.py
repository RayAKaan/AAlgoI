import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import time

from core.context_engine import ContextEngine
from core.problem_spec import ProblemSpec, ProblemType
from core.rl.reward_shaper import RewardShaper, AdaptiveRewardShaper

logger = logging.getLogger(__name__)


class AAlgoIEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, algorithm_registry: Dict, config: Dict = None):
        super().__init__()
        self.algorithm_registry = algorithm_registry
        self.config = config or {}

        self.algorithm_names = list(algorithm_registry.keys())
        self.num_algorithms = len(self.algorithm_names)

        self.state_dim = self.config.get("state_dim", 200)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_algorithms)

        reward_type = self.config.get("reward_shaper", "adaptive")
        self.reward_shaper = AdaptiveRewardShaper(self.config) if reward_type == "adaptive" else RewardShaper(self.config)

        self.context_engine = ContextEngine()
        self.current_problem = None
        self.current_data = None
        self.current_context = None
        self.step_count = 0
        self.episode_count = 0
        self.best_score_episode = 0.0
        self._history_cache = {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        options = options or {}

        if "problem" in options and "data" in options:
            self.current_problem = options["problem"]
            self.current_data = options["data"]
        else:
            self.current_problem, self.current_data = self._sample_problem()

        ctx = self.context_engine.analyze(
            self.current_data,
            task_type=getattr(self.current_problem, "problem_type", "sorting") if hasattr(self.current_problem, "problem_type") else "sorting",
        )
        ctx["time_budget_ms"] = self.config.get("time_budget_ms", 1000)
        ctx["memory_budget_mb"] = self.config.get("memory_budget_mb", 1024)
        ctx["priority"] = self.config.get("priority", "balanced")
        self.current_context = ctx

        self.step_count = 0
        self.best_score_episode = 0.0
        self.episode_count += 1
        state = self._build_state()

        info = {
            "episode": self.episode_count,
            "data_size": len(self.current_data) if hasattr(self.current_data, "__len__") else 0,
            "step": self.step_count,
        }
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        algo_name = self.algorithm_names[action]
        algorithm = self.algorithm_registry[algo_name]
        logger.debug(f"Step {self.step_count}: selected {algo_name}")

        try:
            start = time.perf_counter()
            result = algorithm.process(self.current_data)
            elapsed = time.perf_counter() - start
            is_valid = algorithm.validate_output(self.current_data, result)

            if is_valid:
                quality = self._assess_quality(self.current_data, result, algorithm)
                metrics = {
                    "wall_time_ms": elapsed * 1000,
                    "quality_score": quality,
                    "accuracy": 1.0,
                    "algorithms": [algorithm],
                    "success": True,
                }
                terminated = True
            else:
                metrics = {
                    "wall_time_ms": elapsed * 1000,
                    "quality_score": 0.0,
                    "accuracy": 0.0,
                    "algorithms": [algorithm],
                    "success": False,
                }
                terminated = True

        except Exception as e:
            logger.error(f"Algorithm {algo_name} failed: {e}")
            metrics = {
                "wall_time_ms": 0,
                "quality_score": 0.0,
                "accuracy": 0.0,
                "algorithms": [algorithm],
                "success": False,
                "error": str(e),
            }
            is_valid = False
            terminated = True

        if metrics.get("quality_score", 0) > self.best_score_episode:
            self.best_score_episode = metrics.get("quality_score", 0)

        max_steps = self.config.get("max_steps_per_episode", 50)
        truncated = self.step_count >= max_steps

        reward = self.reward_shaper.compute_reward(
            is_valid=is_valid, metrics=metrics,
            context=self.current_context, history=self._get_history(),
        )

        next_state = self._build_state()

        info = {
            "step": self.step_count,
            "algorithm": algo_name,
            "valid": is_valid,
            "success": metrics.get("success", False),
            "quality": metrics.get("quality_score", 0),
            "time_ms": metrics.get("wall_time_ms", 0),
        }
        if terminated or truncated:
            info["episode_stats"] = {
                "total_reward": reward, "success": metrics.get("success", False),
                "steps": self.step_count, "best_quality": self.best_score_episode,
            }
        return next_state, reward, terminated, truncated, info

    def _build_state(self) -> np.ndarray:
        components = []

        problem_vec = self._infer_problem_features()
        components.append(np.pad(problem_vec, (0, max(0, 128 - len(problem_vec))), "constant")[:128])

        sys_state = np.zeros(16)
        try:
            import psutil
            sys_state[0] = psutil.cpu_percent(interval=0) / 100.0
            sys_state[1] = psutil.cpu_count() / 16.0
            sys_state[2] = psutil.virtual_memory().percent / 100.0
        except Exception:
            pass
        sys_state[10] = self.step_count / max(self.config.get("max_steps_per_episode", 50), 1)
        sys_state[11] = self.best_score_episode
        components.append(sys_state[:16])

        hist_vec = np.zeros(56)
        try:
            if hasattr(self, "_history_cache"):
                if len(self._history_cache) > 0:
                    pass
        except Exception:
            pass
        components.append(hist_vec[:56])

        state = np.concatenate(components)
        target = self.observation_space.shape[0]
        if len(state) < target:
            state = np.pad(state, (0, target - len(state)))
        elif len(state) > target:
            state = state[:target]
        return state.astype(np.float32)

    def _infer_problem_features(self) -> np.ndarray:
        features = np.zeros(128)
        if self.current_context:
            profile = self.current_context.get("data_profile", {})
            features[0] = np.log10(profile.get("size", 1) + 1) / 10.0
            features[1] = 1.0 if profile.get("patterns", {}).get("is_sorted", False) else 0.0
            features[2] = 1.0 if profile.get("patterns", {}).get("is_nearly_sorted", False) else 0.0
            features[3] = profile.get("unique_ratio", 0.5)
            stats = profile.get("statistics", {})
            features[5] = stats.get("mean", 0.0) / 1000.0 if stats.get("mean") else 0.0
            features[6] = stats.get("std", 0.0) / 1000.0 if stats.get("std") else 0.0
            constraints = self.current_context.get("constraints", {})
            features[10] = constraints.get("time_budget_ms", 1000) / 10000.0
            features[11] = constraints.get("memory_budget_mb", 1024) / 10000.0
            priority = constraints.get("priority", "balanced")
            features[15] = 1.0 if priority == "speed" else 0.0
            features[16] = 1.0 if priority == "accuracy" else 0.0
        return features

    def _get_history(self) -> Dict:
        return {"best_score": self.best_score_episode, "algorithm_combinations": []}

    def _assess_quality(self, input_data, output, algorithm) -> float:
        if isinstance(output, (list, np.ndarray)):
            if len(output) <= 1:
                return 1.0
            inversions = sum(1 for i in range(len(output) - 1) if output[i] > output[i + 1])
            return 1.0 - (inversions / max(len(output), 1))
        return 0.95

    def _sample_problem(self) -> Tuple[Any, Any]:
        import random
        size = random.randint(10, 1000)
        data = [random.randint(0, 10000) for _ in range(size)]
        spec = ProblemSpec(name="random_sorting", problem_type=ProblemType.TRANSFORMATION)
        return spec, data

    def render(self):
        pass

    def close(self):
        pass
