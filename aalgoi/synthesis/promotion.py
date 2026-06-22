from __future__ import annotations

from typing import Any

from aalgoi.synthesis.validator import SynthesisValidator


class PromotionManager:
    def __init__(self) -> None:
        self.validator = SynthesisValidator()

    def promote(self, code: str, data: Any = None, expected: Any = None, name: str | None = None) -> dict:
        validation = self.validator.validate_code(code, data=data, timeout=10.0)
        if not validation.get("safe"):
            return {"success": False, "error": validation.get("error", "unsafe code")}
        if expected is not None:
            actual = validation.get("result")
            if actual != expected:
                return {"success": False, "error": f"expected {expected!r}, got {actual!r}"}
        return {"success": True, "message": f"Algorithm '{name or 'unnamed'}' promoted", "code": code}
