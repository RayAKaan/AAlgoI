from __future__ import annotations

from typing import Any

from aalgoi.security.sandbox import Sandbox


class SynthesisValidator:
    def __init__(self) -> None:
        self.sandbox = Sandbox()

    def validate_code(self, code: str, timeout: float = 5.0) -> dict:
        safe, reason = self.sandbox.check_ast_safety(code)
        if not safe:
            return {"safe": False, "error": reason}
        result = self.sandbox.run(code, [], timeout=timeout)
        return {
            "safe": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
        }
