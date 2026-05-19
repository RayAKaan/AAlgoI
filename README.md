# AAlgoI — Adaptive Algorithm Intelligence

**Artificial Algorithm Intelligence** — a self-adaptive system that automatically selects, executes, and learns from the best algorithm for any given problem. Combining reinforcement learning, a semantic knowledge graph, and a growing registry of 20+ algorithms across sorting, pathfinding, optimization, and machine learning domains.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-226%20passing-brightgreen.svg)](tests/)

---

## Features

- **Algorithm Selector** — PPO-based RL agent picks the optimal algorithm from 20+ candidates based on problem type and data patterns
- **Semantic Knowledge Graph** — NetworkX-powered graph encoding relationships (Problem → Algorithm → Pattern → Constraint) that constrains RL choices and provides fallback alternatives
- **Pre-Trained Model** — 3-stage pipeline (supervised + RL curriculum + self-play) produces a 0.72 MB model with ~0.6 ms inference, validated on sorting, pathfinding, and domain routing (100% accuracy)
- **Self-Learning** — Every execution records feedback (quality, timing, success) to personalize the model via fine-tuning and bandit updates
- **Algorithm Marketplace** — Discover, register, benchmark, and share custom algorithms across instances
- **Natural Language Interface** — `question_parser.py` converts English descriptions into structured problem specs
- **Federated Learning** — Optional P2P or central-server mode for cross-instance knowledge sharing
- **Rich CLI** — 10 commands via `click`: `solve`, `benchmark`, `marketplace`, `ml`, `debug`, `web`, `api`, and more
- **226 Passing Tests** — CI-verified test suite

---

## Quick Start

```bash
# Install
pip install aalgoi

# Solve a problem by description
aalgoi solve "sort this list of numbers" --data 3,1,4,1,5,9,2,6,5,3,5

# Solve with explicit spec
aalgoi solve-spec --type sorting --data 5,2,8,1,9

# Start the web UI
aalgoi web

# Start the REST API
aalgoi api
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
pip install aalgoi[web]      # Web UI (Gradio, FastAPI)
pip install aalgoi[llm]      # LLM integration (Ollama)
pip install aalgoi[full]     # Everything
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
Web (optional): gradio, fastapi, uvicorn

---

## Architecture

```
User Input (CLI / Web / API)
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

1. **Input** — CLI command, natural language, or structured API call
2. **Parsing** — `SmartSolver` / `question_parser` extract problem type and data
3. **State Encoding** — 200-dimensional vector: data stats + problem type one-hot + environment info
4. **RL Selection** — PPO agent outputs action probabilities over 20 algorithms
5. **KG Validation** — Knowledge graph constrains the action to semantically valid candidates
6. **Execution** — Algorithm runs on the data; quality, timing, and success are recorded
7. **Feedback** — Results update the bandit, knowledge base, and optionally fine-tune the RL model

---

## Pre-Trained Model

AAlgoI ships with `pretrained_v1.pt` — a 3-stage pre-trained model:

| Stage | Iterations | Description |
|-------|-----------|-------------|
| 1. Supervised Bootstrapping | 200,000 | Textbook rules via CrossEntropy on raw logits |
| 2. RL Curriculum (optional) | 100,000 | PPO refinement with curriculum-based difficulty |
| 3. Self-Play (optional) | — | Adversarial WorldModel (disabled by default) |

### Guarantees

| Check | Requirement | Actual |
|-------|-------------|--------|
| Sorting Accuracy | 100% | 100% |
| Pathfinding Accuracy | 100% | 100% |
| Domain Routing | 100% | 100% |
| Inference Time | < 5 ms | ~0.6 ms |
| Model Size | < 10 MB | 0.72 MB |

Generate your own:

```bash
python training/pretrain_master.py
```

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
| `aalgoi web` | Launch Gradio web UI |
| `aalgoi api` | Start FastAPI REST server |

---

## Adding Algorithms

```python
from pipeline import UniversalSolver
from algorithms.base import Algorithm

class MyCustomSort(Algorithm):
    def process(self, data):
        return sorted(data)

solver = UniversalSolver()
solver.registry["my_sort"] = MyCustomSort()
```

For persistent registration, add your algorithm to `algorithms/` and register it in `pipeline.py`.

---

## Training

```bash
# Quick pre-train (test)
python training/pretrain_master.py

# Full pipeline with all stages
python training/full_train.py

# Distributed training (multi-GPU)
torchrun --nproc_per_node=2 training/distributed_train.py

# Federated training
python training/federated_train.py --mode central --server localhost:5000
```

---

## Tests

```bash
pytest tests/ -v
# 226 passed
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
│   ├── cli.py          # Click CLI (10 commands)
│   ├── cli_ml.py       # ML subcommands
│   ├── cli_debug.py    # Debug subcommands
│   ├── web_ui.py       # Gradio web interface
│   ├── api.py          # FastAPI server
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
