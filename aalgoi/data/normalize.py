from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time as dt_time, timedelta
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


def normalize(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, Path):
        return _load_path(data)
    if isinstance(data, str):
        return _try_parse_string(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "tolist"):
        try:
            return data.detach().cpu().tolist()
        except Exception:
            pass
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, Fraction):
        return float(data)
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, date):
        return data.isoformat()
    if isinstance(data, dt_time):
        return data.isoformat()
    if isinstance(data, timedelta):
        return data.total_seconds()
    if isinstance(data, complex):
        return {"real": data.real, "imag": data.imag}
    if isinstance(data, range):
        return list(data)
    if isinstance(data, Enum):
        return normalize(data.value)
    if is_dataclass(data) and not isinstance(data, type):
        return normalize(asdict(data))
    if isinstance(data, dict):
        return {str(k): normalize(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [normalize(v) for v in data]
    if isinstance(data, set):
        try:
            return sorted(normalize(v) for v in data)
        except Exception:
            return [normalize(v) for v in data]
    if isinstance(data, (int, float, bool)):
        return data
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
        with open(path, encoding="utf-8") as f:
            return normalize(json.load(f))
    if path.suffix in (".csv",):
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            return [normalize(dict(row)) for row in csv.DictReader(f)]
    with open(path, encoding="utf-8") as f:
        return f.read()


def _try_parse_string(s: str) -> Any:
    if s == "" or not s.strip():
        return s
    try:
        return normalize(json.loads(s))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    path = Path(s).expanduser()
    if path.is_file():
        return _load_path(path)
    return s
