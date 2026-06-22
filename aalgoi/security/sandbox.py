from __future__ import annotations

import ast
import multiprocessing
from typing import Any

from aalgoi.errors import UnsafeCode

_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hex", "id", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip",
}

_DANGEROUS_CALLS = {"__import__", "exec", "eval", "compile", "open", "input"}


def check_ast_safety(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise UnsafeCode(f"Syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise UnsafeCode(f"Import not allowed: {node.names[0].name}")
        if isinstance(node, ast.ImportFrom):
            raise UnsafeCode(f"Import not allowed: from {node.module} import ...")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_CALLS:
                raise UnsafeCode(f"Dangerous call not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__") and node.func.attr.endswith("__"):
                raise UnsafeCode(f"Dunder method call not allowed: {node.func.attr}")


class Sandbox:
    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def execute(self, source: str, data: Any = None) -> Any:
        check_ast_safety(source)
        return execute_sandboxed(source, data, self.timeout)


def execute_sandboxed(source: str, data: Any = None, timeout: float = 5.0) -> Any:
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_run_in_child, args=(source, data, child_conn))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.kill()
        proc.join()
        raise UnsafeCode("Execution timed out")
    if parent_conn.poll():
        result = parent_conn.recv()
        if isinstance(result, Exception):
            raise result
        return result
    raise UnsafeCode("No result from sandboxed execution")


def _run_in_child(source: str, data: Any, conn: multiprocessing.connection.Connection) -> None:
    try:
        restricted_globals = {"__builtins__": {name: __builtins__[name] for name in _SAFE_BUILTINS if name in __builtins__}}
        local_vars: dict = {}
        exec(source, restricted_globals, local_vars)
        result = local_vars.get("solve", lambda x: x)(data)
        conn.send(result)
    except Exception as e:
        conn.send(e)
