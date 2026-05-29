# AAlgoI — Adaptive Algorithm Intelligence

**Artificial Algorithm Intelligence** — a self-adaptive system that automatically selects, executes, and learns from the best algorithm for any given problem. Combining reinforcement learning, a semantic knowledge graph, and a growing registry of 20+ algorithms across sorting, pathfinding, optimization, and machine learning domains.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-365%20passing-brightgreen.svg)](tests/)

---

## Features

- **Algorithm Selector** — PPO-based RL agent picks the optimal algorithm from 20+ candidates based on problem type and data patterns
- **Semantic Knowledge Graph** — NetworkX-powered graph encoding relationships (Problem → Algorithm → Pattern → Constraint) that constrains RL choices and provides fallback alternatives
- **Pre-Trained Model** — 3-stage pipeline (supervised + RL curriculum + self-play) produces a 0.72 MB model with ~0.6 ms inference, validated on sorting, pathfinding, and domain routing (100% accuracy)
- **Self-Learning** — Every execution records feedback (quality, timing, success) to personalize the model via fine-tuning and bandit updates
- **Algorithm Marketplace** — Discover, register, benchmark, and share custom algorithms across instances
- **Natural Language Interface** — `question_parser.py` converts English descriptions into structured problem specs
- **Federated Learning** — Optional P2P or central-server mode for cross-instance knowledge sharing
- **Rich CLI** — `solve`, `benchmark`, `marketplace`, `ml`, `debug`, and more
- **365 Passing Tests** — CI-verified test suite

---

## Quick Start

```bash
# Install
pip install aalgoi

# Solve a problem by description
aalgoi solve "sort this list of numbers" --data 3,1,4,1,5,9,2,6,5,3,5

# Solve with explicit spec
aalgoi solve-spec --type sorting --data 5,2,8,1,9
```

---

## Python API

After `pip install aalgoi`, the simplest usage is one line:

```python
from aalgoi import solve

result = solve("sort ascending", [3, 1, 4, 1, 5])
print(result["result"])       # [1, 1, 3, 4, 5]
print(result["algorithm"])    # "insertion_sort"
print(result["time_ms"])      # 1.2
print(result["success"])      # True
```

The `result` dict always has these keys: `result`, `algorithm`, `time_ms`, `success`, `answer` (a human-readable summary from SmartSolver).

### Sorting

```python
from aalgoi import solve

# Basic sort
solve("sort these numbers", [5, 3, 8, 1, 2])
# → algorithm picks insertion_sort, quicksort, timsort, etc.

# With priority on speed
solve("sort as fast as possible", [5, 3, 8, 1, 2])
# → parser detects "fast" → adds objective minimize execution_time

# Large data
solve("sort this list", list(range(10000, 0, -1)))
# → RL agent picks quicksort or timsort for large reverse-sorted data

# Already-sorted data (adaptive handling)
solve("sort this", [1, 2, 3, 4, 5])
# → context_engine detects is_sorted=True, RL picks timsort (O(n))
```

### Pathfinding

```python
from aalgoi import solve

graph = {
    "A": {"B": 5, "C": 2},
    "B": {"D": 4},
    "C": {"B": 1, "D": 7},
    "D": {}
}

# Auto-detects PATHFINDING from data shape
result = solve("find shortest path from A to D", graph)
print(result["algorithm"])    # "dijkstra" or "a_star"
print(result["result"])       # ['A', 'C', 'B', 'D']  (shortest path)

# Explicit question
result = solve("shortest path from A to D", graph)
print(result["algorithm"])    # "dijkstra"
```

### Optimization (Knapsack)

```python
from aalgoi import solve

items = [
    {"value": 60, "weight": 10},
    {"value": 100, "weight": 20},
    {"value": 120, "weight": 30},
]

result = solve("maximize value within capacity 50", {
    "items": items,
    "capacity": 50
})
print(result["algorithm"])              # "greedy_knapsack"
print(result["result"]["selected"])      # [1, 2]
print(result["result"]["value"])         # 220
```

### Using ProblemSpec

```python
from aalgoi import solve_spec, ProblemSpec, ProblemType

spec = ProblemSpec(
    name="custom_sort",
    problem_type=ProblemType.SORTING,
)

result = solve_spec(spec, [3, 1, 2])
print(result["result"])  # [1, 2, 3]
```

### SmartSolver (with explanation)

```python
from aalgoi import SmartSolver

solver = SmartSolver()
result = solver.ask("sort this list quickly", [4, 2, 7, 1])

print(result["answer"])
# "Solved using timsort in 0.57ms."

print(result["algorithm"])  # "timsort"
print(result["time_ms"])    # 0.57
```

### Explaining an algorithm

```python
from aalgoi import explain, solve

result = solve("sort these", [3, 1, 2])
exp = explain(result)          # extracts algorithm name from result
print(exp.summary)
# "Timsort is a stable hybrid sorting algorithm..."

# Direct by name
exp = explain("quicksort")
print(exp.complexity)   # "O(n log n) average, O(n²) worst case"
print(exp.steps)        # ["Choose a pivot...", "Partition...", "Recursively apply..."]
```

### Benchmarking

```python
from aalgoi import benchmark, ProblemSpec, ProblemType

spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
bm = benchmark(spec, [5, 3, 1, 4, 2])
print(bm["winner"])            # "Baseline" (or "AAlgoI")
print(bm["aalgoi_algorithm"])  # "insertion_sort"
```

### Registering a custom algorithm

```python
from aalgoi import UniversalSolver
from algorithms.base import Algorithm

class MySorter(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "my_sorter"
        self.time_complexity = "O(n log n)"

    def process(self, data):
        return sorted(data)

    def validate_output(self, input_data, output_data):
        return all(output_data[i] <= output_data[i+1]
                   for i in range(len(output_data)-1))

solver = UniversalSolver()
solver.register_algorithm(MySorter())

# RL agent now considers it alongside the 20 built-in algorithms
result = solver.solve(ProblemSpec(name="x", problem_type=ProblemType.SORTING), [3, 1, 2])
```

### Quick reference — problem type detection

Each call auto-detects the problem type from the data + question text, selects the optimal algorithm via the PPO policy, executes it, stores the experience for online RL training (every 20 solves), and returns the result.

```python
from aalgoi import solve

sorting       = solve("sort this", [3, 1, 2])
pathfinding   = solve("find path from A to B", {"A": {"B": 1}, "B": {}})
optimization  = solve("maximize value", {"items": [...], "capacity": 10})
clustering    = solve("cluster this data", [[1, 2], [5, 8], [1.5, 1.8]])
word2vec      = solve("train word2vec on medical corpus with 200 dimensions",
                      {"corpus": ["heart disease treatment", "lung cancer diagnosis"]})
image_blur    = solve("blur this image", image_array)
```

---

## Installation

### From PyPI

```bash
pip install aalgoi
```

For optional features:

```bash
pip install aalgoi[rl]       # Reinforcement learning (PyTorch)
pip install aalgoi[llm]      # LLM integration (Ollama)
```

### From Source

```bash
git clone https://github.com/RayAKaan/AAlgoI.git
cd AAlgoI
pip install -e .
```

### Dependencies

Core: numpy, scikit-learn, networkx, chromadb, click, psutil
RL (optional): torch>=2.0.0

---

## Architecture

```
User Input (CLI)
        │
        ▼
┌───────────────────┐
│   SmartSolver     │  Natural language → ProblemSpec
│ question_parser   │  Zero-shot + keyword fallback
└─────────┬─────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│        UniversalMetaController             │
│                                            │
│  ┌──────────┐   ┌──────────────────┐       │
│  │ RL Agent │   │ Knowledge Graph   │       │
│  │ (PPO)    │   │ (read-only filter)│       │
│  │ 20-action│   │ Problem→Algorithm │       │
│  │ softmax  │   │ Algorithm→Pattern │       │
│  └────┬─────┘   │ Algorithm→Complex │       │
│       │         └────────┬─────────┘       │
│       └─────────┬────────┘                  │
│                 ▼                           │
│         Selected Algorithm                  │
│                 │                           │
│  ┌──────────────▼──────────────┐            │
│  │   Algorithm Registry        │            │
│  │   20 algorithms (growing)   │            │
│  └──────────────┬──────────────┘            │
│                 │                           │
│  ┌──────────────▼──────────────┐            │
│  │   Execution + Feedback      │            │
│  │   quality, timing, success  │            │
│  └──────────────┬──────────────┘            │
│                 │                           │
│  ┌──────────────▼──────────────┐            │
│  │   Knowledge Base (ChromaDB) │            │
│  │   Pattern Store + Metrics   │            │
│  └─────────────────────────────┘            │
└────────────────────────────────────────────┘
```

### Pipeline Flow

1. **Input** — CLI command or natural language
2. **Parsing** — `SmartSolver` / `question_parser` extract problem type and data
3. **State Encoding** — 42-dimensional vector: data statistics + problem type one-hot (16 types) + environment info + constraint flags
4. **RL Selection** — PPO agent outputs action probabilities over 20 algorithms
5. **KG Validation** — Knowledge graph constrains the action to semantically valid candidates
6. **Execution** — Algorithm runs on the data; quality, timing, and success are recorded
7. **Feedback** — Results update the bandit, knowledge base, and optionally fine-tune the RL model

---

## Pre-Trained Model

AAlgoI ships with `pretrained_v1.pt` — a 3-stage pre-trained model:

 | Stage | Default Iterations | Description |
 |-------|------------------|-------------|
 | 1. Supervised Bootstrapping | 20,000 | Textbook rules via CrossEntropy on raw logits |
 | 2. RL Curriculum (optional) | 5,000 | PPO refinement with curriculum-based difficulty |
 | 3. Self-Play (optional) | 0 | Adversarial WorldModel (disabled by default) |

### Guarantees

| Check | Requirement | Actual |
|-------|-------------|--------|
| Sorting Accuracy | 100% | 100% |
| Pathfinding Accuracy | 100% | 100% |
| Domain Routing | 100% | 100% |
| Inference Time | < 5 ms | ~0.8 ms |
| Model Size | < 10 MB | 0.41 MB |

Generate your own:

```bash
python training/pretrain_master.py
```

### 42-Dimensional State Vector

The RL agent's state vector encodes all information needed to choose an algorithm:

| Slots | Feature | Description |
|-------|---------|-------------|
| `[0]` | Data size | `log10(n+1) / 7` normalized |
| `[1..3]` | Pattern flags | is_sorted, is_nearly_sorted, is_reverse |
| `[4..6]` | Uniqueness | unique_ratio, has_duplicates, has_negatives |
| `[8..9]` | Distribution | tanh(mean/std), std/mean ratio |
| `[14..15]` | System state | cpu_free, mem_ratio |
| `[18..33]` | Problem type | 16-way one-hot (SORTING, PATHFINDING, OPTIMIZATION, CLUSTERING, CLASSIFICATION, REGRESSION, ML, IMAGE_PROCESSING, SEARCH, NLP, ROUTING, TRANSFORMATION, GENERATION, DECISION, SCHEDULING, COMPUTER_VISION) |
| `[34..39]` | Constraints | speed priority, memory limit, accuracy, scalability, stability, interpretability |
| `[40..41]` | Size flags | large (>100k), small (<100) |

### Attention Head

Instead of a linear policy head, the agent uses **Scaled Dot-Product Attention** to select algorithms:

```
Q = W_q · state       # Query: what kind of algorithm do I need?
K = W_k · algo_emb    # Keys: what does each algorithm offer?
scores = Q · K^T / √d # Affinities: match query to each key
π(a) = softmax(scores)# Policy: probability distribution
```

The attention head learns **relative relationships** — algorithms with embeddings similar to the query vector get higher probability. This naturally generalizes to new algorithms: register a new algorithm and it receives a learnable embedding, then attention compares it against the same query derived from the state. No retraining needed for unseen algorithms; the attention head handles variable-size action spaces.

Each algorithm embedding `eᵢ ∈ ℝ⁶⁴` is learned end-to-end during pre-training and fine-tuned via PPO gradients during RL.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `aalgoi solve` | Solve a problem from natural language description |
| `aalgoi solve-spec` | Solve a structured problem (type, data, constraints) |
| `aalgoi explain` | Explain why a specific algorithm was chosen |
| `aalgoi stats` | Show performance statistics per algorithm |
| `aalgoi benchmark` | Run benchmarks across algorithms |
| `aalgoi marketplace list` | List registered algorithms |
| `aalgoi marketplace search` | Search algorithms by name/pattern |
| `aalgoi ml train-word2vec` | Train a Word2Vec model |
| `aalgoi ml similar-words` | Find similar words in a trained model |
| `aalgoi ml visualize-embeddings` | Visualize word embeddings |
| `aalgoi debug visualize` | Visualize internal state / decision boundaries |

---

## Adding Algorithms

See the **Registering a custom algorithm** section under [Python API](#python-api) above.

For persistent registration, add your algorithm to `algorithms/` and register it in `pipeline.py`.

---

## Training

```bash
# Pre-train the model
python training/pretrain_master.py
```

---

## Tests

```bash
pytest tests/ -v
# 365 passed
```

---

## Project Structure

```
AAlgoI/
├── algorithms/         # Algorithm implementations by domain
│   ├── sorting/        # quicksort, timsort, heapsort, etc.
│   ├── pathfinding/    # dijkstra, a_star, bfs_path
│   ├── optimization/   # greedy_knapsack, simulated_annealing
│   └── ml/             # word2vec, pca, tsne, semantic
├── core/               # Core engine
│   ├── rl/             # PPO agent, WorldModel, reward shaper
│   ├── meta_controller.py  # Central orchestration
│   ├── knowledge_graph.py  # Semantic relationship graph
│   ├── knowledge_base.py   # Vector performance store
│   ├── pipeline_graph.py   # Execution pipeline
│   └── problem_spec.py     # Problem type system
├── interface/          # User interfaces
│   ├── cli.py          # Click CLI
│   ├── cli_ml.py       # ML subcommands
│   ├── cli_debug.py    # Debug subcommands
│   └── nl_parser.py    # Natural language parsing
├── training/           # Training pipelines
│   ├── pretrain_master.py  # 3-stage pre-training
│   ├── self_play.py        # Adversarial self-play
│   ├── curriculum.py       # Difficulty scheduler
│   └── data_generator.py   # Synthetic problem generator
├── tests/              # 226 test cases
├── checkpoints/        # Pretrained model files
└── pipeline.py         # UniversalSolver entry point
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Links

- GitHub: [https://github.com/RayAKaan/AAlgoI](https://github.com/RayAKaan/AAlgoI)
- Issues: [https://github.com/RayAKaan/AAlgoI/issues](https://github.com/RayAKaan/AAlgoI/issues)
