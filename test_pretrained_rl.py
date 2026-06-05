import random
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from aalgoi.core.context_engine import ContextEngine
from aalgoi.core.problem_spec import ProblemSpec, ProblemType
from aalgoi.core.rl import PPOAgent
from aalgoi.pipeline import UniversalSolver

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
    cpu = env_info.get("cpu", {})
    mem = env_info.get("memory", {})
    vec[10] = (100.0 - cpu.get("percent_used", 50)) / 100.0 if cpu else 0.5
    vec[11] = mem.get("available_gb", 1) / max(mem.get("total_gb", 1), 1)
    all_types = list(ProblemType)
    ptype = problem_type
    if ptype in all_types:
        idx = all_types.index(ptype)
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
    print(f"\n  {'='*60}")
    print(f"  Domain: {domain.upper()}")
    print(f"  {'='*60}")
    for spec, data, desc in test_cases:
        state = build_state(data, spec.problem_type)
        action, log_prob, value = agent.select_action(state, deterministic=True)
        algo_name = idx_to_algo[action]
        valid, ms, _ = eval_algo(algo_name, data)
        status = "PASS" if valid else "FAIL"
        print(f"    [{status}] {desc:40s} -> {algo_name:20s} ({ms:6.1f}ms)")

def generate_graph(num_nodes, density=0.3):
    nodes = [chr(ord('A') + i) if i < 26 else f"N{i}" for i in range(num_nodes)]
    graph = {n: {} for n in nodes}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < density:
                weight = random.randint(1, 20)
                graph[nodes[i]][nodes[j]] = weight
    return {"graph": graph, "start": nodes[0], "goal": nodes[-1]}

def generate_knapsack(num_items):
    items = [(random.randint(1, 50), random.randint(1, 100)) for _ in range(num_items)]
    capacity = sum(w for w, _ in items) // 2
    return {"items": items, "capacity": capacity}

for label, ckpt_path in [
    ("Final Model", "checkpoints/pretrain_v2/pretrained_final.pt"),
    ("Best Ep1500", "checkpoints/pretrain_v2/best_model_ep1500.pt"),
    ("Best Ep1000", "checkpoints/pretrain_v2/best_model_ep1000.pt"),
]:
    agent = PPOAgent(state_dim=200, num_actions=len(algo_names), config={})
    agent.load(ckpt_path)
    agent.network.eval()

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {'='*70}")

    run_domain_test("sorting", [
        (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
         [random.randint(0, 1000) for _ in range(100)], "100 random ints"),
        (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
         list(range(1000)), "1K sorted"),
        (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
         list(range(10000, 0, -1)), "10K reverse"),
        (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
         [random.randint(0, 10) for _ in range(50)], "few unique (50)"),
        (ProblemSpec(name="s", problem_type=ProblemType.TRANSFORMATION),
         [random.randint(0, 100000) for _ in range(5000)], "5K random"),
    ])

    run_domain_test("pathfinding", [
        (ProblemSpec(name="p", problem_type=ProblemType.PATHFINDING),
         generate_graph(10, 0.4), "10-node sparse graph"),
        (ProblemSpec(name="p", problem_type=ProblemType.PATHFINDING),
         generate_graph(26, 0.3), "26-node graph"),
        (ProblemSpec(name="p", problem_type=ProblemType.PATHFINDING),
         generate_graph(50, 0.15), "50-node sparse graph"),
        (ProblemSpec(name="p", problem_type=ProblemType.PATHFINDING),
         generate_graph(20, 0.6), "20-node dense graph"),
    ])

    run_domain_test("optimization", [
        (ProblemSpec(name="o", problem_type=ProblemType.OPTIMIZATION),
         generate_knapsack(10), "10-item knapsack"),
        (ProblemSpec(name="o", problem_type=ProblemType.OPTIMIZATION),
         generate_knapsack(20), "20-item knapsack"),
        (ProblemSpec(name="o", problem_type=ProblemType.OPTIMIZATION),
         generate_knapsack(50), "50-item knapsack"),
    ])

print(f"\n{'='*70}")
print("  All tests complete!")
print(f"{'='*70}")
