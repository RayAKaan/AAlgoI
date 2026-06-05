import ast
import builtins
import multiprocessing as mp
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

# ── Security Configuration ──────────────────────────────────────────────

_MP_INITIALIZED = False

def _ensure_mp_context():
    global _MP_INITIALIZED
    if not _MP_INITIALIZED:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        _MP_INITIALIZED = True


SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
    'callable', 'chr', 'complex', 'dict', 'divmod', 'enumerate',
    'filter', 'float', 'frozenset', 'getattr', 'hasattr',
    'hash', 'hex', 'int', 'isinstance', 'issubclass', 'iter',
    'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord',
    'pow', 'range', 'repr', 'reversed', 'round', 'set', 'slice',
    'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
}

DANGEROUS_MODULES = {
    'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
    'shutil', 'pathlib', 'ctypes', 'gc', 'threading',
    'multiprocessing', 'pickle', 'marshal', 'shelve', 'dbm',
    'importlib', 'code', 'codeop', 'pty', 'fcntl',
}

SANDBOX_TIMEOUT = 5.0

# ── AST Validator ───────────────────────────────────────────────────────

class SandboxValidator(ast.NodeVisitor):
    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            top = alias.name.split('.')[0]
            if top in DANGEROUS_MODULES:
                self.errors.append(f"Forbidden import: {alias.name}")
            else:
                self.errors.append(f"Import not allowed: {alias.name}")

    def visit_ImportFrom(self, node):
        top = node.module.split('.')[0]
        if top in DANGEROUS_MODULES:
            self.errors.append(f"Forbidden import from: {node.module}")
        else:
            self.errors.append(f"Import from not allowed: {node.module}")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in ('__import__', 'exec', 'eval', 'compile', 'open'):
                self.errors.append(f"Forbidden call: {attr}")
        if isinstance(node.func, ast.Name):
            if node.func.id in ('exec', 'eval', 'compile', 'open', 'input'):
                self.errors.append(f"Forbidden call: {node.func.id}")
            if node.func.id.startswith('__'):
                self.errors.append(f"Forbidden dunder call: {node.func.id}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith('__') and node.id.endswith('__'):
            self.errors.append(f"Forbidden name: {node.id}")
        self.generic_visit(node)

    def validate(self, source: str) -> bool:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
            return False

        self.visit(tree)
        return len(self.errors) == 0


# ── Sandboxed Module Creator ────────────────────────────────────────────

def create_sandboxed_module(name: str, source: str) -> ModuleType | None:
    validator = SandboxValidator()
    if not validator.validate(source):
        return None

    safe_builtins = {}
    for k in SAFE_BUILTINS:
        if hasattr(builtins, k):
            safe_builtins[k] = getattr(builtins, k)

    restricted_globals = {
        '__builtins__': safe_builtins,
        '__name__': name,
        '__doc__': None,
    }

    module = ModuleType(name)
    try:
        exec(source, restricted_globals, module.__dict__)
    except Exception:
        return None

    module.__source_code__ = source
    return module


# ── Process-Based Timeout Execution ────────────────────────────────────

def _worker(conn, func_source: str, func_name: str, input_data: Any):
    """Execute function in a subprocess with pipe communication."""
    try:
        local_ns = {}
        exec(func_source, local_ns)
        func = local_ns.get(func_name)
        if func is None:
            conn.send((False, "Function not found in source"))
            conn.close()
            return
        result = func(input_data)
        conn.send((True, result))
    except Exception as e:
        conn.send((False, str(e)))
    finally:
        conn.close()


def execute_sandboxed(
    module: ModuleType,
    func_name: str,
    input_data: Any,
    timeout: float = SANDBOX_TIMEOUT,
) -> tuple[bool, Any]:
    func = getattr(module, func_name, None)
    if not callable(func):
        return False, None

    func_source = getattr(module, "__source_code__", None)
    if func_source is None:
        return False, None

    _ensure_mp_context()

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_worker, args=(child_conn, func_source, func_name, input_data))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        if p.is_alive():
            p.kill()
            p.join()
        parent_conn.close()
        return False, None

    if parent_conn.poll():
        try:
            success, payload = parent_conn.recv()
            parent_conn.close()
            if success:
                return True, payload
            else:
                return False, None
        except (EOFError, OSError):
            parent_conn.close()
            return False, None
    else:
        parent_conn.close()
        return False, None


# ── Benchmark Wrapper ───────────────────────────────────────────────────

def benchmark_sandboxed(
    module: ModuleType,
    func_name: str,
    input_data: Any,
    baseline_func: Callable,
    trials: int = 5,
) -> tuple[bool, float, float]:
    for _ in range(2):
        ok, _ = execute_sandboxed(module, func_name, input_data)
        if not ok:
            return False, 0.0, 0.0

    sandbox_times = []
    for _ in range(trials):
        start = time.perf_counter()
        ok, _ = execute_sandboxed(module, func_name, input_data)
        if not ok:
            return False, 0.0, 0.0
        sandbox_times.append(time.perf_counter() - start)

    baseline_times = []
    for _ in range(trials):
        start = time.perf_counter()
        baseline_func(input_data)
        baseline_times.append(time.perf_counter() - start)

    return (
        True,
        sum(sandbox_times) / len(sandbox_times),
        sum(baseline_times) / len(baseline_times),
    )
