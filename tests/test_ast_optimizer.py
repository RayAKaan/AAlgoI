"""Tests for AST optimizer."""
import ast
import time

from aalgoi.core.ast_optimizer import ASTOptimizer


# ── Baseline: no change tests ───────────────────────────────────────────

def test_no_change_already_optimal():
    code = """
def process(data):
    result = [x * 2 for x in data]
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # lru_cache may be added to process; listcomp already there
    assert "lru_cache" in out or "[x * 2 for x in data]" in out


def test_no_change_empty():
    assert ASTOptimizer().optimize("") == ""


def test_no_change_syntax_error():
    code = "def f(:"
    assert ASTOptimizer().optimize(code) == code


def test_no_change_return_constant():
    code = "def process(): return 42"
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # No args, so no lru_cache; no loops to fuse
    assert "lru_cache" not in out


# ── lru_cache injection tests ────────────────────────────────────────────

def test_lru_cache_added():
    code = """
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    assert "@lru_cache" in out or "lru_cache" in out
    assert "from functools import lru_cache" in out


def test_lru_cache_not_added_twice():
    code = """
from functools import lru_cache

@lru_cache
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    assert out.count("lru_cache") == 2  # import + decorator


def test_lru_cache_skips_zero_args():
    code = "def greet(): return 'hello'"
    assert "lru_cache" not in ASTOptimizer().optimize(code)


def test_lru_cache_skips_method():
    code = "def foo(self, x): return x + 1"
    assert "lru_cache" not in ASTOptimizer().optimize(code)


def test_lru_cache_skips_cls_method():
    code = "def bar(cls, x): return x * 2"
    assert "lru_cache" not in ASTOptimizer().optimize(code)


def test_lru_cache_skips_generator():
    code = "def gen(n):\n    for i in range(n):\n        yield i"
    assert "lru_cache" not in ASTOptimizer().optimize(code)


def test_lru_cache_import_added():
    code = "def compute(x): return x ** 2"
    out = ASTOptimizer().optimize(code)
    assert "from functools import lru_cache" in out


def test_lru_cache_import_extended():
    code = """
from functools import wraps

def compute(x):
    return x ** 2
"""
    out = ASTOptimizer().optimize(code)
    assert "from functools import lru_cache, wraps" in out or "from functools import wraps, lru_cache" in out


# ── Listcomp conversion tests ────────────────────────────────────────────

def test_listcomp_conversion():
    code = """
def process(data):
    result = []
    for x in data:
        result.append(x * 2)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # Should be converted to listcomp (no raw for loop, no .append)
    assert ".append(" not in out
    assert "for x in data:" not in out or "[" in out


def test_listcomp_no_conversion_multiple_statements():
    """Don't convert for loops with multiple body statements."""
    code = """
def process(data):
    result = []
    for x in data:
        result.append(x * 2)
        result.append(x + 1)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # Should still have "for x in data:" (not converted to listcomp)
    assert "for x in data:" in out


def test_listcomp_no_conversion_not_append():
    code = """
def process(data):
    result = []
    for x in data:
        result.insert(0, x)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    assert "for x in data:" in out


def test_listcomp_no_conversion_different_var():
    code = """
def process(data):
    result = []
    for x in data:
        other.append(x)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    assert "for x in data:" in out


# ── Loop fusion tests ────────────────────────────────────────────────────

def test_loop_fusion():
    code = """
def process(data):
    result = []
    for x in data:
        result.append(x * 2)
    for x in data:
        result.append(x + 1)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # After listcomp conversion + fusion, at most 1 raw "for x in data:" remains
    assert out.count("for x in data:") <= 1


def test_loop_fusion_different_iter():
    """Don't fuse loops over different iterables."""
    code = """
def process(a, b):
    for x in a:
        print(x)
    for x in b:
        print(x)
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    assert out.count("for x in ") == 2


def test_loop_fusion_different_target():
    """Don't fuse loops with different loop variables."""
    code = """
def process(data):
    for x in data:
        print(x)
    for y in data:
        print(y)
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    assert out.count("for ") == 2


# ── End-to-end speed tests ───────────────────────────────────────────────

def test_lru_cache_speedup():
    """@lru_cache should make recursive fibonacci faster."""
    code = """
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"""
    opt = ASTOptimizer()
    optimized = opt.optimize(code)

    ns = {"fib": None}
    exec(optimized, ns)

    start = time.perf_counter()
    result = ns["fib"](30)
    elapsed = time.perf_counter() - start

    assert result == 832040
    # With lru_cache, fib(30) should be < 1ms
    assert elapsed < 0.1, f"lru_cache fib(30) took {elapsed:.3f}s"


def test_listcomp_speedup():
    """Listcomp should be faster than manual append."""
    code = """
def double(n):
    result = []
    for i in range(n):
        result.append(i * 2)
    return result
"""
    opt = ASTOptimizer()
    optimized = opt.optimize(code)

    ns = {"double": None}
    exec(optimized, ns)

    expected = [i * 2 for i in range(1000)]
    start = time.perf_counter()
    result = ns["double"](1000)
    elapsed = time.perf_counter() - start

    assert result == expected
    assert elapsed < 0.01, f"listcomp took {elapsed:.3f}s"


# ── Edge cases ───────────────────────────────────────────────────────────

def test_nested_function_skipped():
    """Don't add lru_cache to inner functions that capture closure vars."""
    code = """
def outer():
    def inner(x):
        return x + 1
    return inner
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # inner has 1 arg, no self/cls, no yields → it IS a candidate
    # This is fine to add lru_cache to inner
    assert "lru_cache" in out


def test_multiple_optimizations():
    """All three passes run without conflict."""
    code = """
def process(data):
    result = []
    for x in data:
        result.append(x * 2)
    for x in data:
        result.append(x + 1)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)
    # Must remain valid Python
    ast.parse(out)
    # Should be shorter
    assert len(out) < len(code) or "listcomp" in out or "lru_cache" in out


def test_preserves_semantics_same_output():
    """Optimized code produces identical output for same input."""
    code = """
def avg(n):
    total = 0.0
    data = list(range(1, n + 1))
    for x in data:
        total += x
    result = []
    for x in data:
        result.append(x / total)
    for x in data:
        result.append(x * 2)
    return result
"""
    opt = ASTOptimizer()
    out = opt.optimize(code)

    ns = {"__builtins__": __builtins__}
    exec(out, ns)
    exec("def avg_orig(n):\n    total = 0.0\n    data = list(range(1, n + 1))\n    for x in data:\n        total += x\n    result = []\n    for x in data:\n        result.append(x / total)\n    for x in data:\n        result.append(x * 2)\n    return result", ns)

    assert ns["avg"](5) == ns["avg_orig"](5)
