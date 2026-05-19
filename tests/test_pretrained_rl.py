"""Test the pre-trained RL model across all domains."""

import sys
import os
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl import PPOAgent
from core.context_engine import ContextEngine
from core.problem_spec import ProblemSpec, ProblemType
from pipeline import UniversalSolver

solver = UniversalSolver()
context_engine = ContextEngine()
registry = solver._get_global_registry()
algo_names = list(registry.keys())
algo_to_idx = {n: i for i, n in enumerate(algo_names)}
idx_to_algo = {i: n for n, i in algo_to_idx.items()}


def build_state(data, problem_type):
    context = context_engine.analyze(data, problem_type.value)
    vec = np.zeros(200, dtype=np.float32)
    dp = context.get("data_profile", {})
    patterns = dp.get("patterns", {})
    vec[0] = np.log10(dp.get("size", 1) + 1) / 5.0
    vec[1] = 1.0 if patterns.get("is_sorted") else 0.0
    vec[2] = 1.0 if patterns.get("is_nearly_sorted") else 0.0
    env_info = context.get("environment", {})
    vec[10] = (100.0 - env_info.get("cpu", {}).get("percent_used", 50)) / 100.0
    all_types = list(ProblemType)
    if problem_type in all_types:
        idx = all_types.index(problem_type)
        if 20 + idx < 200:
            vec[20 + idx] = 1.0
    return vec


def eval_algo(algo_name, data):
    algo = registry[algo_name]
    try:
        start = time.time()
        result = algo.process(data)
        elapsed = time.time() - start
        valid = algo.validate_output(data, result)
        return valid, elapsed * 1000, result
    except Exception as e:
        return False, 0, str(e)


def run_domain_test(domain, test_cases):
    print(f"\n{'='*60}")
    print(f"Domain: {domain.upper()}")
    print(f"{'='*60}")
    for spec, data, desc in test_cases:
        state = build_state(data, spec.problem_type)
        action, _, _ = agent.select_action(state, deterministic=True)
        algo_name = idx_to_algo[action % len(idx_to_algo)]
        valid, ms, _ = eval_algo(algo_name, data)
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {desc:40s} -> {algo_name:20s} ({ms:6.1f}ms)")


def generate_graph(n, d=0.3):
    nodes = [str(i) for i in range(n)]
    graph = {n: {} for n in nodes}
    for i in range(n):
        for j in range(i + 1, min(i + 3, n)):
            if random.random() < d:
                graph[nodes[i]][nodes[j]] = random.randint(1, 10)
    return {'graph': graph, 'start': nodes[0], 'end': nodes[-1]}


if __name__ == "__main__":
    for label, ckpt_path in [
        ("Final Model", "checkpoints/pretrain_v2/pretrained_final.pt"),
        ("Best Ep1500", "checkpoints/pretrain_v2/best_model_ep1500.pt"),
    ]:
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        agent = PPOAgent(state_dim=200, num_actions=len(algo_names), config={})
        agent.load(ckpt_path)
        agent.network.eval()

        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        run_domain_test("sorting", [
            (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
             [random.randint(0, 1000) for _ in range(100)], "100 random ints"),
            (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
             list(range(1000)), "1K sorted"),
            (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
             list(range(10000, 0, -1)), "10K reverse"),
        ])

        run_domain_test("pathfinding", [
            (ProblemSpec(name="p", problem_type=ProblemType.PATHFINDING),
             generate_graph(10, 0.4), "10-node sparse graph"),
            (ProblemSpec(name="p", problem_type=ProblemType.PATHFINDING),
             generate_graph(26, 0.3), "26-node graph"),
        ])
