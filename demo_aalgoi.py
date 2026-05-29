"""aalgoi 1.4.0 — real-world problem solver demo.
Run:  pip install aalgoi  &&  python demo_aalgoi.py
"""
from aalgoi import solve, solve_spec, sort, path, all_paths, distance
from aalgoi import knapsack, cluster, explain, why, benchmark
from aalgoi import ProblemSpec, ProblemType

print("=" * 60)
print("  AAlgoI — Adaptive Algorithm Intelligence")
print("  Solving real problems in real time")
print("=" * 60)

# ── 1. Sorting ──────────────────────────────────────────────
print("\n\u2500" * 60)
print("  1. SORTING  —  pick the fastest algorithm for your data")
print("\u2500" * 60)

data = [64, 34, 25, 12, 22, 11, 90, 5]
print(f"\n  Input:  {data}")
r = solve("sort ascending", data)
print(f"  Output: {r.result}")
print(f"  Method: {r.algorithm}  ({r.time_ms:.1f} ms)")

# bigger data to see the RL agent pick something smarter
big = [97, 53, 2, 81, 44, 19, 33, 76, 11, 65, 29, 48, 7, 92, 15, 38, 61, 84]
print(f"\n  Input:  {big}")
r = solve("sort this list fast", big)
print(f"  Output: {r.result}")
print(f"  Method: {r.algorithm}  ({r.time_ms:.1f} ms)")

# ── 2. Pathfinding ─────────────────────────────────────────
print("\n\u2500" * 60)
print("  2. PATHFINDING  —  shortest route on a map")
print("\u2500" * 60)

city_map = {
    "Home":  {"School": 5, "Store": 8},
    "School": {"Home": 5, "Library": 2, "Park": 4},
    "Store":  {"Home": 8, "Library": 3, "Gym": 6},
    "Library": {"School": 2, "Store": 3, "Park": 1},
    "Park":  {"School": 4, "Library": 1, "Gym": 2},
    "Gym":   {"Store": 6, "Park": 2},
}

print(f"\n  Map: Home → ... → Gym")
best = path(city_map, "Home", "Gym")
print(f"  Shortest: {' → '.join(best)}")

d = distance(city_map, "Home", "Gym")
print(f"  Distance: {d if d is not None else 'N/A'}")

# ── 3. Knapsack / Optimization ──────────────────────────────
print("\n\u2500" * 60)
print("  3. OPTIMIZATION  —  pack the most valuable items")
print("\u2500" * 60)

inventory = [
    {"name": "Laptop",   "value": 1500, "weight": 2},
    {"name": "Tablet",   "value": 800,  "weight": 1},
    {"name": "Camera",   "value": 900,  "weight": 1.5},
    {"name": "Jacket",   "value": 200,  "weight": 0.5},
    {"name": "Book",     "value": 100,  "weight": 1.5},
    {"name": "Shoes",    "value": 250,  "weight": 1},
    {"name": "Snacks",   "value": 50,   "weight": 0.3},
    {"name": "Charger",  "value": 120,  "weight": 0.2},
]

print(f"\n  Capacity: 4 kg")
result = knapsack(inventory, capacity=4)
sel = result.get("selected", result.get("result", []))
if sel and isinstance(sel[0], int):
    names = [inventory[i]["name"] for i in sel if i < len(inventory)]
    total_v = sum(inventory[i]["value"] for i in sel if i < len(inventory))
    print(f"  Pack:     {names}")
    print(f"  Value:    {total_v}")
elif sel and isinstance(sel[0], dict):
    print(f"  Pack:     {[s['name'] for s in sel]}")
    total_v = sum(s["value"] for s in sel)
    print(f"  Value:    {total_v}")
else:
    print(f"  Pack:     {sel}")
print(f"  Method:   {result.get('algorithm', 'auto')}")

# ── 4. Clustering ───────────────────────────────────────────
print("\n\u2500" * 60)
print("  4. CLUSTERING  —  find natural groups in coordinates")
print("\u2500" * 60)

points = [
    [1.0, 1.0], [1.2, 0.9], [1.1, 1.3],
    [5.0, 5.0], [5.3, 4.8], [5.1, 5.2],
    [9.0, 1.0], [8.8, 0.9], [9.2, 1.2],
]
labels = cluster(points)

print(f"\n  Points: {len(points)} coordinates")
for i, (pt, lbl) in enumerate(zip(points, labels)):
    print(f"    {pt}  → cluster {lbl}")

# ── 5. Explain / Why ───────────────────────────────────────
print("\n\u2500" * 60)
print("  5. EXPLANATION  —  why did it pick that algorithm?")
print("\u2500" * 60)

e = explain("quicksort")
print(f"\n  Algorithm: quicksort")
print(f"  Summary:   {e.summary}")
print(f"  Steps:     {e.complexity}")

r = solve("sort these numbers", [3, 1, 4, 1, 5, 9, 2, 6])
w = why(r)
print(f"\n  Decision explanation:")
print(f"  {w}")

# ── 6. Benchmark vs Standard Library ────────────────────────
print("\n\u2500" * 60)
print("  6. BENCHMARK  —  AAlgoI vs standard library")
print("\u2500" * 60)

spec = ProblemSpec("speed_test", problem_type=ProblemType.SORTING)
data_big = [99, 44, 12, 78, 3, 56, 88, 23, 41, 67, 9, 35]

bm = benchmark(spec, data_big)
print(f"\n  AAlgoI:  {bm.get('solver_time_ms', 0):.2f} ms")
print(f"  Library:  {bm.get('library_time_ms', 0):.2f} ms")
print(f"  Speedup:  {bm.get('speedup', 1):.2f}x")
print(f"  Winner:   {bm.get('winner', '?')}")

# ── 7. Natural Language ────────────────────────────────────
print("\n\u2500" * 60)
print("  7. NATURAL LANGUAGE  —  ask in plain English")
print("\u2500" * 60)

queries = [
    "sort ascending",
    "find the smallest",
]

for q in queries:
    r = solve(q, [3, 1, 4, 1, 5])
    print(f'\n  Question: "{q}"')
    print(f"  Input:    [3, 1, 4, 1, 5]")
    print(f"  Output:   {r.result}")

# ── 8. ProblemSpec (advanced API) ──────────────────────────
print("\n\u2500" * 60)
print("  8. PROBLEM SPEC  —  structured problem definition")
print("\u2500" * 60)

spec = ProblemSpec(
    "custom_sort",
    problem_type=ProblemType.SORTING,
)
r = solve_spec(spec, [7, 2, 9, 1])
print(f"\n  Input:  [7, 2, 9, 1]")
print(f"  Output: {r.result}")
print(f"  Via:    {r.algorithm}")

# ── 9. Stability with edge cases ───────────────────────────
print("\n\u2500" * 60)
print("  9. EDGE CASES  —  handles real-world messiness")
print("\u2500" * 60)

cases = [
    ("empty list", []),
    ("single item", [42]),
    ("already sorted", [1, 2, 3, 4, 5]),
    ("all identical", [7, 7, 7, 7]),
    ("negative numbers", [-5, 3, -1, 0, 2]),
]
for label, data in cases:
    r = solve("sort", data)
    print(f"  {label:20s}  {str(data):30s} → {r.result}  ({r.algorithm})")

# ── Done ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  All done — AAlgoI solved every problem live.")
print("  More:  aalgoi --help   |   aalgoi web")
print("=" * 60)
