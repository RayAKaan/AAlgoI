# aalgoi — Algorithmic AI

[![PyPI version](https://img.shields.io/pypi/v/aalgoi.svg)](https://pypi.org/project/aalgoi/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/aalgoi.svg)](https://pypi.org/project/aalgoi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An algorithmic mind that learns, reasons, and discovers.**

aalgoi solves algorithmic problems from natural language descriptions. It doesn't look up answers — it reasons about problems, selects strategies, generates code, proves correctness, and learns from experience. Over time, it discovers new algorithms.

```python
import aalgoi

result = aalgoi.solve("sort the array", [3, 1, 4, 1, 5])
print(result)          # [1, 1, 3, 4, 5]
print(result.algorithm)  # tim_sort
print(result.complexity) # O(n log n)
result.explain()       # full reasoning trace
```

## Install

```bash
pip install aalgoi
```

Requires Python 3.11+, PyTorch 2.0+, and NetworkX 3.0+. All CPU — no GPU needed.

Optional data format support:

```bash
pip install aalgoi[data]     # numpy, pandas
pip install aalgoi[all]      # numpy, pandas, polars, scipy, scikit-learn, Pillow, pyarrow, pydantic
```

## Quick Start

### One-liner solve

```python
import aalgoi

# Sort
aalgoi.solve("sort the array", [5, 2, 8, 1, 9])
# → [1, 2, 5, 8, 9]

# Search
aalgoi.solve("find target in sorted array", {"nums": [1,3,5,7,9], "target": 7})
# → 3

# GCD
aalgoi.solve("find gcd", {"a": 48, "b": 18})
# → 6

# Two sum
aalgoi.solve("find two numbers that sum to target",
             {"nums": [2, 7, 11, 15], "target": 9})
# → [0, 1]

# Maximum subarray
aalgoi.solve("find maximum sum subarray", [-2, 1, -3, 4, -1, 2, 1, -5, 4])
# → 6
```

### Any data format

```python
import numpy as np
import pandas as pd

# NumPy
aalgoi.solve("find peaks", np.array([1, 3, 2, 5, 4]))

# Pandas DataFrame
aalgoi.solve("predict revenue", pd.read_csv("sales.csv"))

# Files — pass a path, it reads it
aalgoi.solve("analyze", "data/metrics.json")

# Tuples, sets, generators — all work
aalgoi.solve("sort", (3, 1, 2))           # tuple → [1, 2, 3]
aalgoi.solve("sort", {5, 3, 1})           # set → [1, 3, 5]
aalgoi.solve("sort", range(5, 0, -1))     # range → [1, 2, 3, 4, 5]

# Complex numbers, Decimals, datetimes, dataclasses — all normalized
from decimal import Decimal
aalgoi.normalize(Decimal("3.14"))          # → 3.14 (float)

from datetime import datetime
aalgoi.normalize(datetime(2024, 1, 15))    # → "2024-01-15T00:00:00" (ISO string)
```

### The result is transparent

```python
result = aalgoi.solve("sort the array", [3, 1, 4, 1, 5])

# Use it like the raw output
print(result)         # [1, 1, 3, 4, 5]
result[0]             # 1
len(result)           # 5
for x in result: ...  # iterates over [1, 1, 3, 4, 5]
result == [1,1,3,4,5] # True

# Or access metadata
result.algorithm      # "tim_sort"
result.complexity     # "O(n log n)"
result.confidence     # 0.95
result.time_ms        # 142.3
result.is_novel       # False
result.code           # "def solve(data):\n    return sorted(data)"
result.explain()      # Full reasoning trace
```

### Session context

```python
with aalgoi.session() as m:
    m.solve("sort", [3, 1, 2])
    m.solve("find gcd", {"a": 12, "b": 8})
    m.learn("sort", [5, 4, 3], expected=[3, 4, 5])
    print(m.status())
```

## Persistent Mind

```python
mind = aalgoi.Mind("~/my_mind")

# Solve — accumulates experience
mind.solve("sort the array", [3, 1, 4, 1, 5])
mind.solve("find shortest path", graph_data)

# Train — bootstrap from expert demonstrations
mind.train(epochs=10)

# Benchmark — 50-problem suite
report = mind.benchmark()
print(report)
# 📊 Benchmark
# ╔══════════════════════════════════════╗
# ║ Accuracy: 42/50 (84%)               ║
# ║ ──────────────────────────────────── ║
# ║ integers        8/10  ██████████░░ 80% ║
# ║ pathfinding     5/5   ████████████ 100%║
# ║ dynamic_prog    6/8   ███████████░ 75% ║
# ╚══════════════════════════════════════╝

# Checkpoint — save a safe state
mind.checkpoint()

# Rollback — revert to last known-good state
mind.rollback("last_good")

# Inspect knowledge
mind.algorithms    # {"tim_sort": AlgorithmInfo(...), "dijkstra": AlgorithmInfo(...), ...}
mind.principles    # ["optimal_substructure", "greedy_exchange", "divide_conquer", ...]
mind.problems      # ["SORTING", "PATHFINDING", "DYNAMIC_PROGRAMMING", ...]

# Federated learning — share discoveries
mind.share()       # export updates
mind.receive()     # import updates from other minds

# Status
print(mind.status())
# 🧠 Algorithmic Mind
# ╔══════════════════════════════════════╗
# ║ Algorithms:  42                      ║
# ║ Principles:  18                      ║
# ║ Problems:    24                      ║
# ║ Discovered:  3                       ║
# ║ Solved:      127                     ║
# ║ Success rate: 89%                    ║
# ║ Avg time:     98.3ms                 ║
# ╚══════════════════════════════════════╝
```

## Algorithm Coverage

aalgoi knows and can reason about algorithms across these domains:

| Domain | Algorithms |
|--------|-----------|
| **Sorting** | tim_sort, quick_sort, merge_sort, heap_sort, radix_sort, bubble_sort, counting_sort |
| **Searching** | binary_search, linear_search, hash_complement, two_pointer |
| **Pathfinding** | dijkstra, bfs, dfs, a_star, bellman_ford, floyd_warshall |
| **Dynamic Programming** | knapsack_01, coin_change, lcs, lis, edit_distance, matrix_chain |
| **Graph** | kruskal, prim, topological_sort, strongly_connected, bridges, articulation |
| **Flow** | edmonds_karp, ford_fulkerson, dinic |
| **String** | rabin_karp, kmp, boyer_moore, z_algorithm, suffix_array |
| **Math** | sieve_of_eratosthenes, euclidean_gcd, fast_exponentiation, matrix_multiply |
| **Optimization** | gradient_descent, simulated_annealing, genetic_algorithm |
| **ML** | random_forest, kmeans, linear_regression, logistic_regression, text_classifier |
| **NLP** | word_embeddings, text_analysis, generation, retrieval |
| **Image** | edge_detection, segmentation, feature_extraction |

It also **discovers** new algorithms through reasoning and reinforcement learning — these show up with `is_novel=True`.

## How It Works

```
Problem text + Data
       │
       ▼
  ┌─────────────┐
  │  Normalize   │  ← numpy/pandas/torch/files → plain Python
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Understand  │  ← NLP comprehension of problem intent
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Reason      │  ← Knowledge graph: 42 algorithms, 18 principles
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Select      │  ← Meta-controller: cold-start bootstrap → UCB exploration
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Synthesize  │  ← Code generation with principle application
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Prove       │  ← Correctness proof with confidence score
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Execute     │  ← Run code, return result
  └─────┬───────┘
        │
        ▼
  ┌─────────────┐
  │  Learn       │  ← PPO training on trajectory, update KG
  └─────────────┘
```

The knowledge graph connects **algorithms** to **mathematical principles** (optimal substructure, greedy exchange, divide and conquer, etc.) and **problem types**. When the mind encounters a new problem, it reasons about which principles apply, selects or synthesizes an algorithm, proves correctness, and learns from the outcome.

## CLI

```bash
# Solve
aalgoi solve "sort the array" --data '[3,1,4,1,5]'
aalgoi solve "find gcd" --data '{"a": 48, "b": 18}'
aalgoi solve "analyze" --file data.csv --explain --code

# Status
aalgoi status

# Train
aalgoi train --epochs 10

# Benchmark
aalgoi benchmark

# Checkpoints
aalgoi checkpoint save
aalgoi checkpoint list

# Rollback
aalgoi rollback --target last_good

# Federation
aalgoi share
aalgoi receive

# Version
aalgoi version
```

## Data Normalization

aalgoi accepts any data format and normalizes it internally:

```python
from aalgoi import normalize, detect_type

# Primitives — pass through
normalize(42)              # 42
normalize(3.14)            # 3.14
normalize("hello")         # "hello"

# Containers — convert to plain Python
normalize((1, 2, 3))       # [1, 2, 3]       (tuple → list)
normalize({3, 1, 2})       # [1, 2, 3]       (set → sorted list)
normalize({"a": 1})        # {"a": 1}        (dict stays dict)

# Standard library types
from decimal import Decimal
from fractions import Fraction
from datetime import datetime, timedelta
from enum import Enum

normalize(Decimal("3.14"))           # 3.14           (→ float)
normalize(Fraction(22, 7))           # 3.142857...    (→ float)
normalize(datetime(2024, 1, 15))     # "2024-01-15T00:00:00"
normalize(timedelta(hours=2))        # 7200.0         (→ seconds)
normalize(complex(3, 4))             # {"real": 3.0, "imag": 4.0}
normalize(range(5))                  # {"start": 0, "stop": 5, "step": 1}

# Enums
class Color(Enum):
    RED = 1
normalize(Color.RED)                 # 1              (→ value)

# Dataclasses
from dataclasses import dataclass
@dataclass
class Point:
    x: float
    y: float
normalize(Point(1.0, 2.0))           # {"x": 1.0, "y": 2.0}

# Bytes — JSON, CSV, or raw
normalize(b'{"key": 42}')            # {"key": 42}    (JSON decode)
normalize(b'a,b\n1,2\n3,4')         # {"columns": [...], "rows": [...]}  (CSV parse)

# Generators — materialized (capped at 10,000)
normalize(i**2 for i in range(5))    # [0, 1, 4, 9, 16]

# NumPy, Pandas, Torch — if installed
import numpy as np
normalize(np.array([1, 2, 3]))       # [1, 2, 3]

# Type detection
detect_type([1, 2, 3])               # "list(3)"
detect_type({"a": 1})                # "dict(1 keys)"
detect_type(42)                      # "int"
```

**Idempotent:** `normalize(normalize(x)) == normalize(x)` for all inputs.

## AlgorithmInfo

```python
mind = aalgoi.Mind("~/my_mind")
info = mind.algorithms["dijkstra"]

info.name              # "dijkstra"
info.time_complexity   # "O((V+E) log V)"
info.space_complexity  # "O(V)"
info.principles        # ["greedy_exchange", "optimal_substructure"]
info.best_for          # ["shortest_path", "weighted_graph", "single_source"]
info.discovered_by     # "bootstrap" or "mind" (if discovered by reasoning)
info.times_used        # 47
info.times_succeeded   # 44
info.performance       # 0.93

print(info.display())
# 📋 dijkstra
# ╔══════════════════════════════════════╗
# ║ Name:        dijkstra                ║
# ║ Time:        O((V+E) log V)          ║
# ║ Space:       O(V)                    ║
# ║ Principles:  greedy_exchange, ...    ║
# ║ Best for:    shortest_path, ...      ║
# ║ Used:        47x (44 succeeded)      ║
# ║ Performance: 0.93                    ║
# ╚══════════════════════════════════════╝
```

## SolveResult API

```python
result = aalgoi.solve("sort", [3, 1, 2])

# Transparent — use like the raw output
str(result)             # "[1, 2, 3]"
bool(result)            # True
result == [1, 2, 3]     # True
result[0]               # 1
len(result)             # 3
list(result)            # [1, 2, 3]
2 in result             # True
result + [4]            # [1, 2, 3, 4]
int(result)             # TypeError (it's a list)
hash(result)            # hash of output

# Metadata
result.ok               # True (output is not None, no error)
result.output           # [1, 2, 3]
result.algorithm        # "tim_sort"
result.complexity       # "O(n log n)"
result.principle        # "divide_conquer"
result.confidence       # 0.95
result.time_ms          # 142.3
result.is_novel         # False
result.code             # "def solve(data):\n    return sorted(data)"
result.iterations       # 7
result.error            # None

# Explanation
result.explain()        # Multi-line reasoning trace

# Pretty print
repr(result)            # Boxed display with metadata
```

## Architecture

aalgoi ships three packages:

| Package | Purpose |
|---------|---------|
| `aalgoi/` | Public API — solve, session, Mind, normalize, CLI |
| `core/` | Mind engine — RL agent, knowledge graph, synthesis, training, federation, correctness prover |
| `algorithms/` | Algorithm implementations — sorting, pathfinding, ML, NLP, optimization, image processing |

Key components:
- **Knowledge Graph** — 42+ algorithms, 18 mathematical principles, 24 problem types, connected by applicability relations
- **Meta-Controller** — cold-start bootstrap table (maps domains to best-known algorithms) → UCB exploration as experience accumulates
- **Algorithm Synthesizer** — generates new algorithms by composing principles
- **Correctness Prover** — attempts proof of correctness with confidence scoring
- **PPO Training** — online reinforcement learning from solve trajectories
- **Bootstrap Trainer** — learns from expert demonstrations
- **Federated Sync** — share and receive algorithm discoveries between minds

## Requirements

- Python 3.11+
- PyTorch 2.0+
- NetworkX 3.0+

Optional (for data format support):
- numpy, pandas, polars, scipy, scikit-learn, Pillow, pyarrow, pydantic

## License

MIT
