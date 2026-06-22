"""Result class for all solve operations."""

from __future__ import annotations

from typing import Any


class Result:
    """Return type for all AAlgoI solve operations.
    Supports both dict-style and attribute-style access.
    """

    KNOWN_KEYS = {
        "result":      "result",
        "algorithm":   "algorithm",
        "time_ms":     "time_ms",
        "success":     "success",
        "answer":      "answer",
        "error":       "error",
        "confidence":  "confidence",
        "alternatives":"alternatives",
        "metrics":     "metrics",
        "pipeline":    "pipeline",
        "ok":          "success",
        "value":       "result",
        "ms":          "time_ms",
        "algo":        "algorithm",
        "output":      "result",
    }

    def __init__(
        self,
        result: Any = None,
        algorithm: str = "",
        time_ms: float = 0.0,
        success: bool = True,
        answer: str = "",
        error: str = "",
        confidence: float = 0.0,
        alternatives: list | None = None,
        metrics: dict | None = None,
        pipeline: list | None = None,
    ) -> None:
        self.result = result
        self.algorithm = algorithm
        self.time_ms = time_ms
        self.success = success
        self.answer = answer
        self.error = error
        self.confidence = confidence
        self.alternatives = alternatives or []
        self.metrics = metrics or {}
        self.pipeline = pipeline or []

    def __getitem__(self, key: str) -> Any:
        canonical = self.KNOWN_KEYS.get(key)
        if canonical is None:
            raise KeyError(
                f"Result has no field '{key}'. "
                f"Valid keys: {sorted(k for k, v in Result.KNOWN_KEYS.items() if k == v)}"
            )
        return getattr(self, canonical)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list:
        return [k for k, v in self.KNOWN_KEYS.items() if k == v and hasattr(self, k)]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in [
            "result", "algorithm", "time_ms", "success",
            "answer", "error", "confidence", "alternatives",
            "metrics", "pipeline",
        ] if k in ("result", "algorithm", "time_ms", "success") or getattr(self, k) is not None}

    @property
    def ok(self) -> bool:
        return self.success

    @property
    def value(self) -> Any:
        return self.result

    @property
    def ms(self) -> float:
        return self.time_ms

    @property
    def algo(self) -> str:
        return self.algorithm

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        conf = f" ({self.confidence:.0%})" if self.confidence > 0 else ""
        return f"Result({status} {self.algorithm}{conf} in {self.time_ms:.2f}ms)"

    def __str__(self) -> str:
        if self.answer:
            return self.answer
        return repr(self)

    def __bool__(self) -> bool:
        return self.success

    def __lt__(self, other: Any) -> bool:
        return self.time_ms < other.time_ms
