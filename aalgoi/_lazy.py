"""
aalgoi/_lazy.py
Thread-safe lazy import utility.

Usage:
    _load_pipeline = lazy_import("pipeline")
    solver = _load_pipeline().UniversalSolver()

    ProblemSpec = lazy_attr("core.problem_spec", "ProblemSpec")
"""
import importlib
import threading

_lock = threading.Lock()
_cache = {}


def lazy_import(module_path: str):
    """Returns a loader function that imports module_path
    on first call and caches it forever."""
    def _load():
        with _lock:
            if module_path not in _cache:
                _cache[module_path] = importlib.import_module(module_path)
            return _cache[module_path]
    return _load


def lazy_attr(module_path: str, attribute: str):
    """Returns the attribute from module_path, loading both lazily."""
    def _load():
        with _lock:
            if module_path not in _cache:
                _cache[module_path] = importlib.import_module(module_path)
            return _cache[module_path]
    return getattr(_load(), attribute)
