# aalgoi — Algorithmic AI (v2.2)

[![PyPI version](https://img.shields.io/pypi/v/aalgoi.svg)](https://pypi.org/project/aalgoi/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/aalgoi.svg)](https://pypi.org/project/aalgoi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An algorithmic mind that learns, reasons, and discovers.**

aalgoi solves algorithmic problems from natural language descriptions. It uses a grounded architecture: parse → retrieve candidates → rank → execute → validate → learn. RL selects among validated algorithms — it does not hallucinate unvalidated code.

> **v2.2** — Complete architectural refactor: typed contracts, canonical algorithm registry, unified Knowledge Graph (NetworkX + SQLite), per-task oracles, contextual bandit learner, AST-safety sandbox (no eval()), and a 99/99 (100%) core benchmark across 33 task types. No PyTorch required.

```python
import aalgoi

result = aalgoi.solve("sort the array", [3, 1, 4, 1, 5])
print(result)            # [1, 1, 3, 4, 5]
print(result.algorithm)  # tim_sort
print(result.validated)  # True
```

## Install

```bash
pip install aalgoi
```

Requires Python 3.11+, NetworkX 3.0+, NumPy 1.24+, scikit-learn 1.2+. All CPU — no GPU needed. PyTorch is optional (`pip install aalgoi[torch]`).

## Quick Start

### One-liner solve

```python
import aalgoi

# Sort
aalgoi.solve("sort the list", [5, 2, 8, 1, 9])
# → [1, 2, 5, 8, 9]

# Search
aalgoi.solve("binary search for target", {"data": [1,3,5,7,9], "target": 7})
# → 3

# GCD
aalgoi.solve("compute gcd", {"a": 48, "b": 18})
# → 6

# Two sum
aalgoi.solve("find two sum", {"data": [2, 7, 11, 15], "target": 9})
# → [0, 1]

# Maximum subarray
aalgoi.solve("kadane maximum subarray", [-2, 1, -3, 4, -1, 2, 1, -5, 4])
# → 6

# Fibonacci (fast doubling, not Binet)
aalgoi.solve("fibonacci number", {"n": 71})
# → 308061521170129

# lcm(0,0) — no divide-by-zero
aalgoi.solve("compute lcm", {"a": 0, "b": 0})
# → 0

# Anagram with normalization
aalgoi.solve("check anagram", {"s1": "Dormitory", "s2": "Dirty room"})
# → True

# Shortest path (weighted)
aalgoi.solve("shortest path weighted dijkstra", {
    "graph": {"A": {"B": 10, "C": 1}, "C": {"B": 1}},
    "start": "A", "end": "B"
})
# → {'path': ['A', 'C', 'B'], 'length': 2}

# Maximum flow
aalgoi.solve("maximum flow", {
    "graph": {"s": {"a": 3, "b": 2}, "a": {"t": 2}, "b": {"t": 3}},
    "source": "s", "sink": "t"
})
# → {'flow_value': 4}

# KMP string matching
aalgoi.solve("kmp pattern search", {
    "text": "abcxabcdabxabcdabcdabcy",
    "pattern": "abcdabcy"
})
# → 15
```

### Shortcuts (one-function-per-task)

```python
from aalgoi.shortcuts import sort, search, knapsack, cluster

sort([3, 1, 2])                        # [1, 2, 3]
search([1, 2, 3], 2)                   # 1
knapsack([{"weight": 10, "value": 60},
          {"weight": 20, "value": 100},
          {"weight": 30, "value": 120}], 50)
# → {'selected': [1, 2], 'value': 220, 'weight': 50}
```

### Persistent Mind

```python
mind = aalgoi.Mind("~/my_mind")

# Solve — accumulates experience
mind.solve("sort the list", [3, 1, 4, 1, 5])
mind.solve("shortest path unweighted", graph_data)

# Benchmark — 99-problem core suite
report = mind.benchmark()
print(report)

# Train — contextual bandit learning
mind.train(mode="bandit", suite="core-v1")

# Checkpoint — save state
mind.checkpoint()

# Rollback — revert to last known-good state
mind.rollback("last_good")

# Inspect knowledge
mind.algorithms    # 42 registered algorithms
mind.principles    # 17 principles (divide_conquer, dynamic_programming, ...)
mind.problems      # 33 problem types

# Status
print(mind.status())
```

## Algorithm Coverage

aalgoi ships 42 algorithms across 33 task types, all validated with per-task oracles:

| Domain | Tasks |
|--------|-------|
| **Sorting** | sort, counting_sort |
| **Searching** | linear_search, binary_search, two_sum, lower_bound |
| **Math** | gcd, lcm, is_prime, sieve, fast_exponentiation, fibonacci |
| **Strings** | palindrome, anagram, kmp, rabin_karp, edit_distance, lcs |
| **Graph** | bfs, dfs, shortest_path_unweighted, shortest_path_weighted (Dijkstra), shortest_path_negative (Bellman-Ford), topological_sort, cycle_detection, connected_components, mst (Kruskal), max_flow (Edmonds-Karp) |
| **DP** | kadane, knapsack_01, coin_change, lis |
| **Optimization** | knapsack_fractional |
| **ML** (optional) | knn_classifier, linear_regression, kmeans |
| **NLP** (optional) | sentiment_analyzer, text_summarizer |
| **Image** (optional) | gaussian_blur, edge_detection |

Optional algorithms require `scikit-learn`, `transformers`, or `Pillow` respectively.

## Architecture

The solve pipeline is fully deterministic and grounded:

```
Problem text + Data
       │
       ▼
  ┌──────────────┐
  │  Parse        │  ← regex + data-based task inference
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Retrieve     │  ← Knowledge Graph (NetworkX + SQLite)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Rank         │  ← RuleRanker + UCB1Bandit (contextual)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Execute      │  ← Sandboxed (AST-check + multiprocess timeout)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Validate     │  ← Per-task oracles verify correctness
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Record       │  ← SQLite run history + bandit reward update
  └──────────────┘
```

- **No eval()** — code safety uses AST analysis + multiprocess sandbox with restricted globals
- **No torch required** — core works without PyTorch; RL uses UCB1 bandit (optional PPO via `aalgoi[torch]`)
- **No unvalidated results** — synthesis is gated behind validation; never returns unvalidated code as success

## Module Structure

| Module | Purpose |
|--------|---------|
| `aalgoi/` | Public API — solve, Mind, session, CLI, shortcuts |
| `aalgoi/algorithms/` | 42 algorithm classes with `@algorithm` decorator, typed specs |
| `aalgoi/types.py` | Core contracts — ProblemTask, ProblemSpec, AlgorithmSpec, SolveResult, BenchmarkReport |
| `aalgoi/problems/` | Parser, per-task oracles, generators, examples |
| `aalgoi/kg/` | Knowledge Graph (NetworkX), SQLite store, query engine, updater |
| `aalgoi/selection/` | Planner, RuleRanker, UCB1Bandit, feature extractor |
| `aalgoi/security/` | AST sandbox, security policies |
| `aalgoi/eval/` | Benchmark suite, metrics, reports |
| `aalgoi/mind/` | Mind orchestration, checkpointing, session, learning |
| `aalgoi/rl/` | Environment, policy, trainer, replay buffer, reward shaper |
| `aalgoi/synthesis/` | Template manager, synthesis validator, promotion manager |
| `aalgoi/data/` | Data normalization, type/shape detection, file loaders |

## CLI

```bash
# Solve
aalgoi solve "sort the list" --data '[3,1,4,1,5]'
aalgoi solve "compute gcd" --data '{"a": 48, "b": 18}'
aalgoi solve "check if prime" --data '{"n": 17}' --explain

# Registry
aalgoi registry

# Knowledge Graph
aalgoi kg stats
aalgoi kg inspect --algorithm tim_sort

# Benchmark
aalgoi benchmark --verbose

# Train
aalgoi train --mode bandit --suite core-v1

# Version
aalgoi version
```

## Data Normalization

```python
from aalgoi.data import normalize, detect_type

normalize((1, 2, 3))       # [1, 2, 3]
normalize({3, 1, 2})       # [1, 2, 3]
normalize(b'{"key": 42}')  # {"key": 42}
detect_type([1, 2, 3])     # "list(3)"
```

## Requirements

- Python 3.11+
- NetworkX 3.0+
- NumPy 1.24+
- scikit-learn 1.2+

Optional:
- `pip install aalgoi[torch]` — PyTorch 2.0+ for PPO training
- `pip install aalgoi[data]` — pandas, polars, pyarrow
- `pip install aalgoi[nlp]` — transformers, gensim, chromadb
- `pip install aalgoi[all]` — everything

## Repository Structure

```
aalgoi/               # Package root
aalgoi/algorithms/    # 42 algorithm implementations + registry
aalgoi/problems/      # Parser, oracles, generators
aalgoi/kg/            # Knowledge Graph + SQLite store
aalgoi/selection/     # Planner, ranker, bandit
aalgoi/security/      # AST sandbox, policies
aalgoi/eval/          # Benchmark suite, metrics
aalgoi/mind/          # Mind, session, checkpoint
aalgoi/rl/            # RL environment, policy, trainer
aalgoi/synthesis/     # Template manager, validator
aalgoi/data/          # Data normalization, detectors
experimental/         # Legacy code (v1/v2) quarantined for reference
```

## License

MIT
