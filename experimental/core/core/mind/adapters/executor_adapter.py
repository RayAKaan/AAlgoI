from typing import Any


class ExecutorAdapter:
    def __init__(self, solver: Any = None) -> None:
        self._solver = solver

    def execute(self, algorithm_name: str, data: Any) -> Any:
        if self._solver is None:
            return None
        try:
            result = self._solver.execute_algorithm(algorithm_name, data)
            return result
        except Exception:
            return None

    def execute_code(self, code: str, data: Any) -> Any:
        try:
            from aalgoi.core.sandbox import execute_sandboxed
            return execute_sandboxed(code, data)
        except ImportError:
            return self._safe_exec(code, data)
        except Exception:
            return None

    def _safe_exec(self, code: str, data: Any) -> Any:
        restricted_globals = {
            "__builtins__": {
                "list": list, "dict": dict, "set": set,
                "len": len, "range": range, "enumerate": enumerate,
                "sorted": sorted, "min": min, "max": max, "sum": sum,
                "abs": abs, "int": int, "float": float, "str": str,
                "bool": bool, "isinstance": isinstance,
                "True": True, "False": False, "None": None,
            }
        }
        local_vars = {}
        try:
            exec(code, restricted_globals, local_vars)
            for val in local_vars.values():
                if callable(val) and not isinstance(val, type):
                    return val(data)
        except Exception:
            pass
        return None
