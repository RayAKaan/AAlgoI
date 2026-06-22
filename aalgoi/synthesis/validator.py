from __future__ import annotations

from typing import Any

from aalgoi.errors import UnsafeCode
from aalgoi.security.sandbox import Sandbox, check_ast_safety


class SynthesisValidator:
    def __init__(self) -> None:
        self.sandbox = Sandbox()

    def validate_code(self, code: str, data: Any = None, timeout: float = 5.0) -> dict:
        try:
            check_ast_safety(code)
        except UnsafeCode as e:
            return {"safe": False, "error": str(e)}
        try:
            sandbox = Sandbox(timeout=timeout)
            result = sandbox.execute(code, data)
            return {
                "safe": True,
                "result": result,
                "stdout": "",
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
            }
        except UnsafeCode as e:
            return {"safe": False, "error": str(e)}
        except Exception as e:
            return {"safe": False, "error": str(e)}
