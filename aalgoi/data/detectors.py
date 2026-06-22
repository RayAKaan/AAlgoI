from __future__ import annotations

from typing import Any

import numpy as np


def detect_type(data: Any) -> str:
    if isinstance(data, np.ndarray):
        return "ndarray"
    if isinstance(data, list):
        return "list"
    if isinstance(data, dict):
        return "dict"
    if isinstance(data, str):
        return "str"
    if isinstance(data, int):
        return "int"
    if isinstance(data, float):
        return "float"
    if isinstance(data, bool):
        return "bool"
    if data is None:
        return "NoneType"
    if hasattr(data, "shape"):
        return "array_like"
    return type(data).__name__


def detect_shape(data: Any) -> str:
    if isinstance(data, np.ndarray):
        return str(data.shape)
    if isinstance(data, list):
        if not data:
            return "(0,)"
        return f"({len(data)},)"
    if isinstance(data, dict):
        return f"({len(data)} keys)"
    return "scalar"
