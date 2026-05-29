# AAlgoI v1.4.0 — Cheat Sheet

```bash
pip install aalgoi
```

---

## One function, any problem

```python
from aalgoi import solve

solve("sort ascending", [3, 1, 4, 1, 5])
# → [1, 1, 3, 4, 5]   using timsort (0.6 ms)

solve("shortest path", graph, "A", "D")
# → ['A', 'B', 'D']

solve("cluster these", points)
# → [0, 0, 1, 1, 2, 2]
```

<details>
<summary><b>Result object</b> — <code>.result</code>  <code>.algorithm</code>  <code>.time_ms</code>  <code>.ok</code></summary>

```python
r = solve("sort", [3, 1, 2])
r.result     # [1, 2, 3]
r.algorithm  # "timsort"
r.time_ms    # 0.57
r.ok         # True
r.algo       # alias
r.ms         # alias
r["result"]  # dict-style too
```
</details>

<details>
<summary><b>Shortcuts</b> — one-liners for common tasks</summary>

```python
from aalgoi import sort, path, knapsack, cluster, search, why

sort([3, 1, 4, 1, 5])                         # → [1, 1, 3, 4, 5]
sort([3, 1, 4], reverse=True)                 # → [4, 3, 1]

path(graph, "A", "D")                         # → ['A', 'B', 'D']
distance(graph, "A", "D")                     # → 3

knapsack(items, capacity=50)                  # → {selected: [...], value: 160}

cluster(points)                               # → [0, 0, 1, 1, 2]

search([1, 2, 3, 4, 5], 3)                   # → 2

why(solve("sort", [3, 1, 2]))                # "Chose insertion_sort..."
explain("quicksort")                          # human-readable summary

benchmark(spec, data)                         # head-to-head vs stdlib
```
</details>

<details>
<summary><b>CLI</b></summary>

```bash
aalgoi solve "sort asc" 3 1 4 1 5
aalgoi e quicksort                     # explain
aalgoi b sort --n 10000                # benchmark
aalgoi web                             # Gradio UI
aalgoi api                             # REST server
```
</details>

<details>
<summary><b>Advanced</b> — custom algorithms, persistent solver</summary>

```python
from aalgoi import SmartSolver, solve_spec, ProblemSpec, ProblemType

solver = SmartSolver()
r1 = solver.ask("sort", [3, 1, 2])

spec = ProblemSpec("my_task", problem_type=ProblemType.SORTING)
r = solve_spec(spec, [4, 2, 7, 1])

from pipeline import UniversalSolver
class MySorter(Algorithm):
    name = "my_sorter"
    def process(self, data): return sorted(data)
solver = UniversalSolver()
solver.register_algorithm(MySorter())
```
</details>

## Version

```python
from aalgoi import __version__   # → "1.4.0"
```

---

**More:** [github.com/RayAKaan/AAlgoI](https://github.com/RayAKaan/AAlgoI)
