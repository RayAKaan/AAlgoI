from __future__ import annotations

from typing import Any

from aalgoi.synthesis.validator import SynthesisValidator


class PromotionManager:
    def __init__(self) -> None:
        self.validator = SynthesisValidator()

    def promote(self, code: str, name: str | None = None) -> dict:
        validation = self.validator.validate_code(code, timeout=10.0)
        if not validation.get("safe"):
            return {"success": False, "error": validation.get("error", "unsafe code")}
        if validation.get("returncode", -1) != 0:
            return {"success": False, "error": f"runtime error: {validation.get('stderr', '')}"}
        return {"success": True, "message": f"Algorithm '{name or 'unnamed'}' promoted", "code": code}
