import torch
import numpy as np
import time
import logging
import os
import json
import random
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any

from aalgoi.core.rl.agents.selection_agent import PPOAgent
from aalgoi.core.context_engine import ContextEngine
from aalgoi.core.problem_spec import ProblemSpec, ProblemType
from aalgoi.core.validator import LearningValidator
from aalgoi.pipeline import UniversalSolver
from training.data_generator import SyntheticDataGenerator, CurriculumLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.algo_selection_counts = defaultdict(int)
        self.domain_win_rates = defaultdict(lambda: deque(maxlen=100))
        self.curriculum_progression = []
        self.best_reward = -float('inf')

    def log_episode(self, reward: float, success: bool, algo_name: str, domain: str):
        self.episode_rewards.append(reward)
        self.success_rates.append(1.0 if success else 0.0)
        self.algo_selection_counts[algo_name] += 1
        self.domain_win_rates[domain].append(1.0 if success else 0.0)
        if reward > self.best_reward:
            self.best_reward = reward

    def summary(self) -> Dict:
        return {
            'avg_reward_last_100': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'success_rate_last_100': float(np.mean(self.success_rates)) if self.success_rates else 0.0,
            'best_reward_ever': self.best_reward,
            'top_algorithms': sorted(
                self.algo_selection_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            'domain_win_rates': {k: float(np.mean(v)) for k, v in self.domain_win_rates.items()},
            'total_episodes': sum(self.algo_selection_counts.values()),
        }


class RLPretrainer:
    def __init__(self, save_dir="checkpoints/pretrain_v2", config: Dict = None):
        self.config = config or {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.solver = UniversalSolver()
        self.agent = self.solver.meta_controller.rl_agent
        self.context_engine = self.solver.context_engine
        self.validator = LearningValidator()
        self.data_gen = SyntheticDataGenerator()

        self.registry = self.solver._get_global_registry()
        self.algo_names = list(self.registry.keys())
        self.algo_to_idx = {name: i for i, name in enumerate(self.algo_names)}
        self.idx_to_algo = {i: name for name, i in self.algo_to_idx.items()}

        self.metrics = TrainingMetrics()
        self.episodes_per_level = self.config.get('episodes_per_level', 800)
        self.current_level = CurriculumLevel.BEGINNER
        self.best_val_reward = -float('inf')
        self.time_budget_ms = self.config.get('time_budget_ms', 500.0)

        logger.info(
            f"RLPretrainer initialized: {len(self.algo_names)} algorithms, "
            f"save_dir={save_dir}"
        )

    def run_training(self, total_episodes: int = 5000):
        logger.info(f"Starting pre-training for {total_episodes} episodes...")

        for episode in tqdm(range(total_episodes), desc="Training RL"):
            if episode > 0 and episode % self.episodes_per_level == 0:
                self._advance_curriculum()

            spec, data = self._generate_problem()
            context = self.context_engine.analyze(data, spec.problem_type.value)
            state = self._build_state_vector(context, spec)

            action_idx, log_prob, value = self.agent.select_action(state, deterministic=False)
            algo_name = self.idx_to_algo[action_idx % len(self.idx_to_algo)]

            reward, success, _ = self._execute_and_reward(algo_name, data, spec, context)

            self.agent.store_transition(state, action_idx, reward, False, log_prob, value)
            self.metrics.log_episode(reward, success, algo_name, spec.problem_type.value)

            if len(self.agent.buffer) >= self.agent.batch_size:
                train_stats = self.agent.train()
                if train_stats and episode % 100 == 0:
                    logger.debug(f"Train loss: {train_stats['total_loss']:.4f}")

            if episode > 0 and episode % 500 == 0:
                self._validate_and_checkpoint(episode)

        self._save_final_model()
        logger.info("Pre-training complete!")
        summary = self.metrics.summary()
        logger.info(json.dumps(summary, indent=2))
        return summary

    def _generate_problem(self) -> Tuple[ProblemSpec, Any]:
        domain = random.choice(['sorting', 'pathfinding', 'optimization'])
        if domain == 'sorting':
            return self.data_gen.generate_sorting()
        elif domain == 'pathfinding':
            return self.data_gen.generate_pathfinding()
        return self.data_gen.generate_optimization()

    def _advance_curriculum(self):
        current_success = float(np.mean(self.metrics.success_rates)) if self.metrics.success_rates else 0.0
        if current_success > 0.75 and self.current_level < CurriculumLevel.EXPERT:
            self.current_level += 1
            self.data_gen.set_level(self.current_level)
            self.metrics.curriculum_progression.append({
                'episode': len(self.metrics.episode_rewards),
                'level': self.current_level,
                'success_rate': current_success,
            })
            logger.info(f"Curriculum advanced to level {self.current_level}")

    def _execute_and_reward(self, algo_name: str, data: Any,
                            spec: ProblemSpec, context: Dict) -> Tuple[float, bool, Dict]:
        algo = self.registry[algo_name]
        prepared_data = self._prepare_data(spec, data)

        try:
            start_time = time.time()
            result = algo.process(prepared_data)
            elapsed = time.time() - start_time

            is_valid = algo.validate_output(prepared_data, result)
            if not is_valid:
                return -10.0, False, {'error': 'validation_failed'}

            quality = self._compute_quality_score(result, spec)
            speed = self._compute_speed_bonus(elapsed)
            total = quality + speed
            return total, True, {'elapsed_ms': elapsed * 1000, 'quality': quality, 'speed': speed}

        except Exception as e:
            return -15.0, False, {'error': str(e)}

    def _prepare_data(self, spec: ProblemSpec, raw_data: Any) -> Any:
        if spec.problem_type == ProblemType.PATHFINDING:
            if isinstance(raw_data, tuple) and len(raw_data) == 3:
                return {'graph': raw_data[0], 'start': raw_data[1], 'end': raw_data[2]}
        elif spec.problem_type == ProblemType.OPTIMIZATION:
            if isinstance(raw_data, tuple) and len(raw_data) == 2:
                return {'items': raw_data[0], 'capacity': raw_data[1]}
        return raw_data

    def _compute_quality_score(self, result: Any, spec: ProblemSpec) -> float:
        if spec.problem_type in (ProblemType.SORTING, ProblemType.TRANSFORMATION):
            if isinstance(result, list) and len(result) > 1:
                if all(result[i] <= result[i + 1] for i in range(len(result) - 1)):
                    return 10.0
            return 0.0
        elif spec.problem_type == ProblemType.PATHFINDING:
            return 10.0 if isinstance(result, list) and len(result) > 0 else 0.0
        elif spec.problem_type == ProblemType.OPTIMIZATION:
            if isinstance(result, dict) and 'selected' in result:
                return 10.0
            return 0.0
        return 5.0

    def _compute_speed_bonus(self, elapsed: float) -> float:
        budget_sec = self.time_budget_ms / 1000.0
        if elapsed < budget_sec:
            return 5.0 * (1.0 - elapsed / budget_sec)
        return -5.0 * (elapsed / budget_sec - 1.0)

    def _build_state_vector(self, context: Dict, spec: ProblemSpec) -> np.ndarray:
        vec = np.zeros(200, dtype=np.float32)
        dp = context.get('data_profile', {})
        patterns = dp.get('patterns', {})

        vec[0] = np.log10(dp.get('size', 1) + 1) / 5.0
        vec[1] = 1.0 if patterns.get('is_sorted') else 0.0
        vec[2] = 1.0 if patterns.get('is_nearly_sorted') else 0.0

        env = context.get('environment', {})
        cpu = env.get('cpu', {})
        mem = env.get('memory', {})
        vec[10] = (100.0 - cpu.get('percent_used', 50)) / 100.0 if cpu else 0.5
        vec[11] = mem.get('available_gb', 1) / max(mem.get('total_gb', 1), 1)

        all_types = list(ProblemType)
        if spec.problem_type in all_types:
            idx = all_types.index(spec.problem_type)
            if 20 + idx < 200:
                vec[20 + idx] = 1.0

        return vec

    def _validate_and_checkpoint(self, episode: int):
        logger.info(f"Running validation at episode {episode}...")
        val_rewards = []

        for _ in range(50):
            spec, data = self._generate_problem()
            context = self.context_engine.analyze(data, spec.problem_type.value)
            state = self._build_state_vector(context, spec)

            action_idx, _, _ = self.agent.select_action(state, deterministic=True)
            algo_name = self.idx_to_algo[action_idx % len(self.idx_to_algo)]

            reward, success, _ = self._execute_and_reward(algo_name, data, spec, context)
            val_rewards.append(reward if success else -10.0)

        avg_val_reward = float(np.mean(val_rewards))
        logger.info(f"Validation Avg Reward: {avg_val_reward:.2f}")

        if avg_val_reward > self.best_val_reward:
            self.best_val_reward = avg_val_reward
            path = os.path.join(self.save_dir, f"best_model_ep{episode}.pt")
            self.agent.save(path)
            logger.info(f"New best model saved: {path}")

    def _save_final_model(self):
        final_path = os.path.join(self.save_dir, "pretrained_final.pt")
        self.agent.save(final_path)

        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.metrics.summary(), f, indent=2)

        logger.info(f"Final model: {final_path}")
        logger.info(f"Summary: {summary_path}")
