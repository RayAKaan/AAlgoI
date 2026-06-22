from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def normalize(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (list, tuple, set)):
        return list(data)
    if isinstance(data, dict):
        return {str(k): normalize(v) for k, v in data.items()}
    if isinstance(data, (int, float, str, bool)):
        return data
    if isinstance(data, Path):
        return _load_path(data)
    if isinstance(data, str):
        return _try_parse_string(data)
    return data


def normalize_with_metadata(data: Any) -> dict:
    result = normalize(data)
    return {
        "data": result,
        "type": type(data).__name__,
        "length": len(result) if isinstance(result, (list, dict)) else None,
    }


def _load_path(path: Path) -> Any:
    if path.suffix in (".json",):
        with open(path) as f:
            return json.load(f)
    if path.suffix in (".csv",):
        import csv
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    with open(path) as f:
        return f.read()


def _try_parse_string(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    path = Path(s)
    if path.exists():
        return _load_path(path)
    return s
