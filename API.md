
# AAlgoI API Documentation

## Core Classes

### AAlgoI (pipeline.py)
Main orchestrator class.

```python
system = AAlgoI(config={
    "strategy": "hybrid",          # Selection strategy
    "time_budget_ms": 500,        # Time budget in milliseconds
    "priority": "balanced",       # Priority: speed, accuracy, balanced
    "retrain_interval": 100,      # Retrain ML model every N executions
    "knowledge_path": "knowledge.json",
    "max_knowledge_size": 100000
})
```

#### Methods

**run(data, task_type="auto", expected_result=None)**
Execute the adaptive pipeline on input data.

- `data`: Input data to process
- `task_type`: Task domain ("sorting", "image_processing", "ml")
- `expected_result`: Optional expected result for quality scoring
- Returns: Processed result

**get_stats()**
Get comprehensive system statistics.

**explain_decision()**
Get human-readable explanation for the last algorithm selection.

**benchmark(data, algorithms=None)**
Benchmark all or specified algorithms on given data.

### ContextEngine (core/context_engine.py)
Analyzes input data and environment.

```python
engine = ContextEngine(config={...})
context = engine.analyze(data, task_type="sorting")
```

Returns context dict with:
- `data_profile`: size, type, statistics, patterns
- `environment`: CPU, memory, disk info
- `constraints`: time budget, priority
- `predictions`: recommended parallelism, risk factors
- `features`: normalized feature vector for ML

### MetaController (core/meta_controller.py)
Selects optimal algorithms using multiple strategies.

**Strategies:**
- `rule-based`: Human-crafted heuristics
- `ml-based`: Trained classifier
- `genetic`: Evolutionary algorithm
- `hybrid`: Combines all strategies

### DynamicCompositor (core/compositor.py)
Modifies algorithms at runtime.

**Modifications:**
- Parametric tuning (threads, quality, buffer size)
- Memoization injection (LRU cache)
- Parallelization (for chunk-safe algorithms)
- AST-based code optimization

### PerformanceTracker (core/performance_tracker.py)
Measures execution performance.

Tracks:
- Wall time and CPU time
- Memory usage
- Quality score (compared to expected result)
- Budget compliance
- Throughput

### KnowledgeBase (core/knowledge_base.py)
Persistent storage for historical decisions.

Features:
- Cosine similarity search
- Algorithm performance statistics
- Best algorithm recommendation
- Auto-pruning when size exceeds limit

## Algorithm Base Class

```python
from algorithms.base import Algorithm

class MyAlgorithm(Algorithm):
    name = "my_algo"
    tags = ["sorting", "fast"]
    complexity = {"time": "O(n log n)", "space": "O(n)"}
    performance_profiles = {
        "large_random": {"score": 0.95, "conditions": {"data_size": ">1000"}}
    }

    def process(self, data):
        # Your algorithm implementation
        return result
```

## Configuration Options

```json
{
  "system_name": "AAlgoI",
  "version": "1.0.0",
  "default_strategy": "hybrid",
  "context_engine": {
    "time_budget_ms": 500,
    "memory_budget_mb": 1024,
    "accuracy_target": 0.95,
    "priority": "balanced"
  },
  "meta_controller": {
    "retrain_interval": 100,
    "exploration_rate": 0.2,
    "min_history_for_ml": 50
  },
  "compositor": {
    "cache_size": 256,
    "max_parallelism": 8,
    "enable_ast_optimization": true
  },
  "performance_tracker": {
    "track_memory": true,
    "track_cpu": true
  },
  "knowledge_base": {
    "path": "knowledge.json",
    "max_size": 100000,
    "auto_save": true
  }
}
```

## Command Line Interface

```bash
# Run demo
python main.py --mode demo

# Benchmark algorithms
python main.py --mode benchmark --size 10000

# Learning demonstration
python main.py --mode learning

# Interactive mode
python main.py --mode interactive
```

## Extending AAlgoI

### Adding New Algorithms

1. Create algorithm class in appropriate domain module
2. Inherit from `Algorithm` base class
3. Implement `process()` method
4. Add to registry in `pipeline.py`

### Adding New Domains

1. Create new module in `algorithms/`
2. Add domain mapping in `MetaController._domain_map`
3. Add detection logic in `ContextEngine`

### Custom Scoring

Override `_compute_composite_score()` in `AAlgoI` to customize
how algorithm performance is evaluated.
