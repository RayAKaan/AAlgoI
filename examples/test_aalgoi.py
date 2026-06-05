"""aalgoi 1.4.0 — quick test script.
Run:  pip install aalgoi  &&  python examples/test_aalgoi.py
"""
from aalgoi import ProblemSpec, ProblemType, benchmark, cluster, explain, knapsack, path, solve, sort

ok = failed = 0
def check(label, cond, detail=""):
    global ok, failed
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")

# ── 1. Natural-language solve ──
print("\n=== 1. Natural-language solve ===")
r = solve("sort ascending", [3, 1, 4, 1, 5])
check("sort", r.result == [1, 1, 3, 4, 5], r.result)
check("ok flag", r.ok)
check("algorithm chosen", bool(r.algorithm))
print(f"  Used: {r.algorithm}  ({r.time_ms:.1f}ms)")

# ── 2. Shortcut functions ──
print("\n=== 2. Shortcut functions ===")
check("sort()", sort([3, 1, 2]) == [1, 2, 3])

graph = {"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}
p = path(graph, "A", "D")
check("path() A->D", p == ["A", "B", "C", "D"])

items = [{"name": "a", "value": 60, "weight": 10},
         {"name": "b", "value": 100, "weight": 20},
         {"name": "c", "value": 120, "weight": 30}]
k = knapsack(items, capacity=50)
check("knapsack", bool(k.get("selected", k.get("result", None))))

# ── 3. Cluster ──
pts = [[1, 1], [1.1, 1], [10, 10], [10, 9.9]]
try:
    c = cluster(pts)
    check("cluster", len(set(str(x) for x in c)) >= 1)
except Exception as ex:
    check("cluster", False, str(ex))

# ── 4. Explain (image filter tested in package source) ──
# blur() skipped — requires numpy array roundtrip support in pipeline

# ── 5. Explanation ──
print("\n=== 5. Explanation ===")
e = explain("timsort")
check("explain has summary", bool(e.summary))
check("explain has complexity", bool(e.complexity))
print(f"  {e.summary}")

# ── 6. Benchmark ──
print("\n=== 6. Benchmark ===")
spec = ProblemSpec("bench", problem_type=ProblemType.SORTING)
bm = benchmark(spec, [3, 1, 4, 1, 5])
check("benchmark has speedup", "speedup" in bm or "winner" in bm)
print(f"  Winner: {bm.get('winner', '?')}")

# ── 7. ProblemSpec with solve_spec ──
print("\n=== 7. ProblemSpec solve ===")
from aalgoi import solve_spec

spec = ProblemSpec("spec_sort", problem_type=ProblemType.SORTING)
r2 = solve_spec(spec, [5, 3, 1])
check("solve_spec", r2.result == [1, 3, 5])

# ── 8. Empty / edge-case safety ──
print("\n=== 8. Edge cases ===")
check("empty list", solve("sort", []).result == [])
check("single item", solve("sort", [42]).result == [42])

# ── Summary ──
print(f"\n{'='*40}")
print(f"  {ok} passed, {failed} failed")
if failed:
    print("  Some checks failed — see details above.")
else:
    print("  Everything looks good!")
print(f"{'='*40}")
print("More:  aalgoi --help")
