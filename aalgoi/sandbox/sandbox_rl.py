"""
aalgoi.sandbox.sandbox_rl — SandboxRL: user-trainable RL for algorithm selection.

SandboxRL gives users a dynamic, trainable policy that lives alongside
the production PPOAgent without touching it.

  - Multiple independent instances (each is a separate A2C policy)
  - Train via curriculum, custom problems, or self-exploration
  - Solve problems while learning from every outcome (auto_learn mode)
  - Fork an instance at any training state to branch specialization
  - Save, load, and resume from any checkpoint
  - Inspect what the policy has learned in plain language

NOTE: In v1, the RL observes and learns from SmartSolver's decisions.
In v2, confident RL predictions will bypass SmartSolver entirely.
Confidence threshold for bypass: agent.epsilon < 0.05 AND confidence > 0.85
"""

from __future__ import annotations
import copy
import os
import time
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from .policy      import PolicyAgent
from .state_bridge import StateBridge
from .reward_bridge import RewardBridge
from .curriculum  import get_problems, CURRICULUM
from .instance_store import register, get_instance, list_instances, delete_instance


class SandboxRL:
    """
    User-trainable RL policy for algorithm selection.

    Quick start:
        from aalgoi.sandbox import SandboxRL

        rl = SandboxRL()
        rl.curriculum()
        result = rl.solve("sort this", {"data": [3,1,2]})
        rl.save("my.rl")

    Multiple instances:
        nlp = SandboxRL(name="nlp", domains=["NLP"])
        opt = SandboxRL(name="opt", domains=["OPTIMIZATION"])
        SandboxRL.list()        # ["nlp", "opt"]
        SandboxRL.get("nlp")    # returns the nlp instance

    Forking:
        base = SandboxRL()
        base.curriculum(end_at=10)
        specialist = base.fork("specialist")
        specialist.teach("my custom problem", data)
    """

    def __init__(
        self,
        name:          Optional[str] = None,
        hidden_size:   int   = 64,
        num_layers:    int   = 2,
        learning_rate: float = 0.003,
        gamma:         float = 0.95,
        epsilon:       float = 0.2,
        auto_learn:    bool  = False,
        domains:       Optional[List[str]] = None,
    ):
        from aalgoi.core.smart_solver import SmartSolver
        self._smart_solver = SmartSolver()
        self._bridge  = StateBridge(self._smart_solver.solver.registry)
        self._reward  = RewardBridge()

        self._agent = PolicyAgent(
            input_size     = self._bridge.state_size,
            num_algorithms = len(self._bridge.algo_names),
            hidden_size    = hidden_size,
            num_layers     = num_layers,
            learning_rate  = learning_rate,
            gamma          = gamma,
            epsilon        = epsilon,
        )

        self.name        = name
        self.domains     = domains
        self.auto_learn  = auto_learn
        self._paused     = False

        self._train_log:  List[dict] = []
        self._solve_log:  List[dict] = []
        self._config = {
            "hidden_size":   hidden_size,
            "num_layers":    num_layers,
            "learning_rate": learning_rate,
            "gamma":         gamma,
        }

        if name:
            register(name, self)

    # ─────────────────────────────────────────────────────────────────────
    # Training methods
    # ─────────────────────────────────────────────────────────────────────

    def curriculum(
        self,
        start_from: int  = 1,
        end_at:     int  = 25,
        rounds:     int  = 1,
        verbose:    bool = True,
    ):
        problems = get_problems(start_from, end_at, self.domains)
        if not problems:
            if verbose:
                print("  No problems match the current domain filter.")
            return

        for rnd in range(rounds):
            if verbose and rounds > 1:
                print(f"\n  Round {rnd+1}/{rounds}")

            for p in problems:
                reward, algo_chosen = self._run_episode(p)
                loss = self._agent.update()

                record = {
                    "problem_id": p["id"],
                    "difficulty": p["difficulty"],
                    "reward":     reward,
                    "loss":       loss,
                    "round":      rnd + 1,
                    "algo":       algo_chosen,
                }
                self._train_log.append(record)

                if verbose:
                    mark = "+" if reward > 0.5 else "-"
                    q    = p["query"][:38]
                    print(f"  [{p['id']:02d}] diff={p['difficulty']} "
                          f'"{q:<38}" {mark} r={reward:+.2f}  {algo_chosen}')

        if verbose:
            self._print_summary()

    def teach(
        self,
        query:          str,
        data:           Any,
        expected_algo:  Optional[str] = None,
        rounds:         int  = 5,
        verbose:        bool = True,
    ):
        problem = {
            "id": 0, "difficulty": 3,
            "query": query, "data": data,
            "expected_type": None,
            "expected_algo": expected_algo,
            "validator": lambda out, p: out is not None,
        }

        for r in range(rounds):
            reward, algo = self._run_episode(problem, expected=expected_algo)
            loss = self._agent.update()
            self._train_log.append({
                "problem_id": 0, "difficulty": 3,
                "reward": reward, "loss": loss,
                "round": r+1, "algo": algo,
                "custom_query": query[:40],
            })

            if verbose:
                mark = "+" if reward > 0.5 else "-"
                print(f"  Round {r+1}/{rounds} {mark} r={reward:+.2f}  {algo}")

    def self_teach(self, rounds: int = 100, verbose: bool = True):
        for r in range(rounds):
            problem  = self._random_problem()
            reward, algo = self._run_episode(problem)
            self._agent.update()
            self._train_log.append({
                "problem_id": 0, "difficulty": 3,
                "reward": reward, "algo": algo, "self_teach": True,
            })

            if verbose and (r + 1) % 10 == 0:
                recent = np.mean([e["reward"] for e in self._train_log[-10:]])
                print(f"  Round {r+1:4d}/{rounds}  "
                      f"avg_r(last 10)={recent:+.3f}  e={self._agent.epsilon:.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # Solve (use SmartSolver, RL observes and learns)
    # ─────────────────────────────────────────────────────────────────────

    def solve(self, query: str, data: Any, learn: bool = True) -> Any:
        """
        Solve a problem using SmartSolver.

        The RL agent predicts which algorithm SmartSolver will choose.
        If learn=True and auto_learn is enabled, the RL records the
        outcome and updates its policy — improving its predictions
        over time.

        Returns the same Result object as SmartSolver.ask().
        """
        state = self._bridge.encode(query, data)
        action_idx, confidence = self._agent.select(state)
        rl_algo = self._bridge.idx_to_algo(action_idx)

        t0 = time.time()
        try:
            result = self._smart_solver.ask(query, data)
            success = result.get("success", False)
            pipeline_algo = result.get("algorithm", "")
            time_ms = (time.time() - t0) * 1000
        except Exception as e:
            result = None
            success = False
            pipeline_algo = ""
            time_ms = (time.time() - t0) * 1000

        if learn and self.auto_learn and not self._paused:
            prediction_match = (rl_algo == pipeline_algo)
            reward = self._reward.compute(
                success=success and prediction_match,
                time_ms=time_ms,
                difficulty=3,
                chosen_algo=rl_algo,
                expected_algo=pipeline_algo,
                confidence=confidence,
            )
            self._agent.record(state, action_idx, reward)

            if len(self._agent._buffer) % 16 == 0:
                self._agent.update()

            self._solve_log.append({
                "query":   query[:40],
                "rl_algo": rl_algo,
                "pipeline_algo": pipeline_algo,
                "success": success,
                "reward":  reward,
            })

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        self._agent.save(path)

    @classmethod
    def from_checkpoint(
        cls,
        path:          str,
        step:          Optional[int] = None,
        name:          Optional[str] = None,
        learning_rate: float = 0.003,
        epsilon:       Optional[float] = None,
        **kwargs,
    ) -> "SandboxRL":
        agent  = PolicyAgent.from_file(path, lr=learning_rate)

        if step is not None:
            agent.episodes = min(step, agent.episodes)

        if epsilon is not None:
            agent.epsilon = epsilon

        instance = cls.__new__(cls)
        from aalgoi.core.smart_solver import SmartSolver
        instance._smart_solver = SmartSolver()
        instance._bridge    = StateBridge(instance._smart_solver.solver.registry)
        instance._reward    = RewardBridge()
        instance._agent     = agent
        instance.name       = name
        instance.domains    = kwargs.get("domains", None)
        instance.auto_learn = kwargs.get("auto_learn", False)
        instance._paused    = False
        instance._train_log = []
        instance._solve_log = []
        instance._config    = {
            "hidden_size":   agent.net.hidden_size,
            "num_layers":    sum(1 for m in agent.net.trunk if hasattr(m, "weight")),
            "learning_rate": learning_rate,
            "gamma":         agent.gamma,
        }

        if name:
            register(name, instance)

        return instance

    # ─────────────────────────────────────────────────────────────────────
    # Forking
    # ─────────────────────────────────────────────────────────────────────

    def fork(self, new_name: Optional[str] = None) -> "SandboxRL":
        forked = SandboxRL.__new__(SandboxRL)
        from aalgoi.core.smart_solver import SmartSolver
        forked._smart_solver = SmartSolver()
        forked._bridge    = StateBridge(forked._smart_solver.solver.registry)
        forked._reward    = RewardBridge()
        forked._agent     = copy.deepcopy(self._agent)
        forked.name       = new_name
        forked.domains    = copy.copy(self.domains)
        forked.auto_learn = self.auto_learn
        forked._paused    = False
        forked._train_log = []
        forked._solve_log = []
        forked._config    = copy.copy(self._config)

        if new_name:
            register(new_name, forked)

        return forked

    # ─────────────────────────────────────────────────────────────────────
    # Inspection
    # ─────────────────────────────────────────────────────────────────────

    def inspect(self, top_k: int = 10):
        s = self._agent.stats()

        name_tag = f" ({self.name})" if self.name else ""
        print(f"\n  {'='*60}")
        print(f"  SandboxRL policy summary{name_tag}")
        print(f"  {'='*60}")
        print(f"  Episodes:      {s['episodes']}")
        print(f"  Total reward:  {s['total_reward']}")
        print(f"  Avg reward:    {s['avg_reward']}")
        print(f"  Recent avg:    {s['recent_avg']}")
        print(f"  Epsilon:       {s['epsilon']}")
        print(f"  Auto-learn:    {self.auto_learn}")
        print(f"  {'='*60}")
        print(f"  Query -> preferred algorithm:")
        print(f"  {'-'*60}")

        problems = get_problems(1, 25)[:top_k]
        for p in problems:
            state = self._bridge.encode(p["query"], p["data"] if p["data"] != "PANDEMIC_DATASET" else {})
            idx, conf = self._agent.select(state)
            algo = self._bridge.idx_to_algo(idx)
            mark = "+" if conf > 0.3 else "~"
            q    = p["query"][:28]
            a    = algo[:18]
            print(f"  {mark} \"{q:<28}\" -> {a:<18} c={conf:.2f}")

        print(f"  {'='*60}\n")

    def stats(self) -> dict:
        agent_stats = self._agent.stats()
        log_rewards = [e["reward"] for e in self._train_log]
        return {
            **agent_stats,
            "name":           self.name,
            "problems_seen":  len(self._train_log),
            "solve_calls":    len(self._solve_log),
            "auto_learn":     self.auto_learn,
            "domains":        self.domains,
            "curriculum_avg": float(np.mean(log_rewards)) if log_rewards else 0.0,
            "config":         self._config,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Controls
    # ─────────────────────────────────────────────────────────────────────

    def pause_learning(self):
        self._paused = True

    def resume_learning(self):
        self._paused = False

    def set_epsilon(self, epsilon: float):
        self._agent.epsilon = float(np.clip(epsilon, 0.0, 1.0))

    def set_lr(self, lr: float):
        self._agent.set_lr(lr)

    def reset(self):
        cfg = self._config
        self._agent = PolicyAgent(
            input_size     = self._bridge.state_size,
            num_algorithms = len(self._bridge.algo_names),
            hidden_size    = cfg["hidden_size"],
            num_layers     = cfg["num_layers"],
            learning_rate  = cfg["learning_rate"],
            gamma          = cfg["gamma"],
        )
        self._train_log.clear()
        self._solve_log.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Class-level instance management
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def list(cls) -> List[str]:
        return list_instances()

    @classmethod
    def get(cls, name: str) -> Optional["SandboxRL"]:
        return get_instance(name)

    @classmethod
    def delete(cls, name: str) -> bool:
        return delete_instance(name)

    # ─────────────────────────────────────────────────────────────────────
    # Internal episode runner
    # ─────────────────────────────────────────────────────────────────────

    def _run_episode(
        self,
        problem:  dict,
        expected: Optional[str] = None,
    ) -> Tuple[float, str]:
        data = problem["data"]
        if data == "PANDEMIC_DATASET":
            data = {}

        state      = self._bridge.encode(problem["query"], data)
        action_idx, confidence = self._agent.select(state)
        algo_name  = self._bridge.idx_to_algo(action_idx)

        success = False
        time_ms = 0.0
        t0 = time.time()
        try:
            algo = self._smart_solver.solver.registry.get(algo_name)
            if algo is not None and data:
                output  = algo.process(data)
                time_ms = (time.time() - t0) * 1000
                validator = problem.get("validator", lambda o, p: o is not None)
                success = bool(validator(output, problem))
            else:
                time_ms = (time.time() - t0) * 1000
        except Exception:
            time_ms = (time.time() - t0) * 1000

        reward = self._reward.compute(
            success      = success,
            time_ms      = time_ms,
            difficulty   = problem.get("difficulty", 3),
            chosen_algo  = algo_name,
            expected_algo= expected or problem.get("expected_algo"),
            confidence   = confidence,
        )

        self._agent.record(state, action_idx, reward)
        return reward, algo_name

    def _random_problem(self) -> dict:
        algo_name = np.random.choice(self._bridge.algo_names)
        algo      = self._smart_solver.solver.registry.get(algo_name)
        meta      = algo.describe() if hasattr(algo, "describe") else {}
        ptypes    = meta.get("problem_types", [])
        ptype     = ptypes[0] if ptypes else "UNKNOWN"

        templates = {
            "SORTING":   ("sort data ascending", {"data": np.random.randint(1,100,20).tolist()}),
            "PATHFINDING": ("find shortest path", {
                "graph": {"S":{"A":1,"B":3},"A":{"T":2},"B":{"T":1},"T":{}}, "start":"S","end":"T"}),
            "OPTIMIZATION": ("maximize value within weight limit", {
                "items": [{"name":f"i{i}","weight":i+1,"value":(i+1)*2} for i in range(8)],
                "capacity": 20}),
            "CLASSIFICATION": ("classify data", {
                "X_train": np.random.randn(30,4).tolist(),
                "y_train": ["A"]*15+["B"]*15,
                "X_test":  np.random.randn(5,4).tolist()}),
            "REGRESSION": ("predict continuous values", {
                "X_train": np.random.randn(30,3).tolist(),
                "y_train": np.random.randn(30).tolist(),
                "X_test":  np.random.randn(5,3).tolist()}),
            "CLUSTERING": ("cluster data into groups", {
                "data": np.random.randn(24,3).tolist(), "n_clusters":3}),
            "NLP": ("analyze text sentiment", {"texts": ["Good product","Bad service","OK overall"]}),
            "IMAGE_PROCESSING": ("detect edges in image", {
                "image": np.random.rand(32,32).tolist()}),
        }

        query, data = templates.get(ptype, ("solve this problem", {}))
        return {"id":0,"difficulty":3,"query":query,"data":data,
                "expected_type":ptype,"validator":lambda o,p:o is not None}

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _print_summary(self):
        rewards = [e["reward"] for e in self._train_log]
        if not rewards:
            return
        passed = sum(1 for r in rewards if r > 0.5)
        print(f"\n  Summary: {passed}/{len(rewards)} problems positive reward  "
              f"|  avg={np.mean(rewards):+.3f}  "
              f"|  e={self._agent.epsilon:.4f}")

    def __repr__(self) -> str:
        s = self._agent.stats()
        name_part = f" name={self.name!r}" if self.name else ""
        return (f"SandboxRL({name_part} episodes={s['episodes']} "
                f"avg_r={s['avg_reward']:.3f} e={s['epsilon']:.3f})")
