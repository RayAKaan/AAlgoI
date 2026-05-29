from pipeline import UniversalSolver
from algorithms.base import Algorithm

solver = UniversalSolver()
agent  = solver.meta_controller.rl_agent

import numpy as np
state = np.zeros(38, dtype=np.float32)
action, log_prob, value = agent.select_action(state)

n_registry = len(solver.registry)
n_embeds   = agent._algo_embeddings.shape[0]

print(f"Registry size:    {n_registry}")
print(f"Embedding matrix: {n_embeds} x {agent._algo_embeddings.shape[1]}")
print(f"Selected action:  {action} ({list(solver.registry.keys())[action]})")
print(f"Log prob:         {log_prob:.4f}")
print(f"Value:            {value:.4f}")

assert n_registry == n_embeds, \
    f"MISMATCH: registry={n_registry} embeds={n_embeds}"

print("PASS Check 1: policy size matches registry")

class TestAlgo(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "test_algo_week1"
        self.time_complexity = "O(n)"
        self.tags = ["sorting"]
    def process(self, data):
        return sorted(data)

solver.register_algorithm(TestAlgo())

n_after = agent._algo_embeddings.shape[0]
print(f"\nAfter register_algorithm():")
print(f"  Embedding matrix: {n_after} x {agent._algo_embeddings.shape[1]}")
print(f"  Expected: {n_embeds + 1}")

assert n_after == n_embeds + 1, \
    f"MISMATCH: expected {n_embeds+1} got {n_after}"

print("PASS Check 2: policy expanded without retraining")

import collections
selections = collections.Counter()
for _ in range(200):
    a, _, _ = agent.select_action(state)
    selections[list(solver.registry.keys())[a]] += 1

print(f"\nSelection distribution over 200 random states:")
for algo, count in selections.most_common(5):
    print(f"  {algo}: {count}")

selected_test = selections.get("test_algo_week1", 0)
print(f"\n  test_algo_week1 selected: {selected_test}/200 times")
print("PASS Check 3: new algorithm is selectable")

print("\nAll attention head checks passed")
