"""
Universal data normalizer.

Converts any input data (numpy, pandas, torch, files, URLs, etc.)
into plain Python types that the mind can process.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from collections import OrderedDict
from dataclasses import fields as dataclass_fields
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any


def normalize(data: Any, allow_file_read: bool = False, allow_url_fetch: bool = False) -> Any:
    """
    Normalize any data to plain Python types.

    Idempotent: normalize(normalize(x)) == normalize(x)

    Security: file/URL auto-detection is DISABLED by default.
    Set allow_file_read=True or allow_url_fetch=True to enable.
    """
    if data is None:
        return None

    # ── Primitives ────────────────────────────────────────────────
    if isinstance(data, bool):
        return data
    if isinstance(data, int):
        return data
    if isinstance(data, float):
        return data
    if isinstance(data, str):
        return _normalize_string(data, allow_file_read=allow_file_read, allow_url_fetch=allow_url_fetch)

    # ── Bytes ─────────────────────────────────────────────────────
    if isinstance(data, bytes):
        return _normalize_bytes(data)

    # ── Containers ────────────────────────────────────────────────
    if isinstance(data, dict):
        return {str(k): normalize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize(item) for item in data]
    if isinstance(data, tuple):
        return [normalize(item) for item in data]
    if isinstance(data, frozenset):
        return sorted([normalize(item) for item in data])
    if isinstance(data, set):
        return sorted([normalize(item) for item in data])
    if isinstance(data, OrderedDict):
        return {str(k): normalize(v) for k, v in data.items()}

    # ── Stdlib types ──────────────────────────────────────────────
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, Fraction):
        return float(data)
    if isinstance(data, complex):
        return {"real": data.real, "imag": data.imag}
    if isinstance(data, range):
        return {"type": "range", "start": data.start, "stop": data.stop, "step": data.step}
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, date):
        return data.isoformat()
    if isinstance(data, dt_time):
        return data.isoformat()
    if isinstance(data, timedelta):
        return data.total_seconds()

    # ── Enum ──────────────────────────────────────────────────────
    if isinstance(data, Enum):
        return data.value

    # ── Dataclass ─────────────────────────────────────────────────
    if _is_dataclass(data):
        return {f.name: normalize(getattr(data, f.name)) for f in dataclass_fields(data)}

    # ── Pydantic (model_dump avoids iteration as field-name list) ─
    if hasattr(data, "model_dump"):
        return normalize(data.model_dump())

    # ── External libraries — BEFORE generic iterable ──────────────
    np = _try_import_numpy()
    if np is not None and isinstance(data, np.ndarray):
        return _normalize_numpy(data, np)

    pd = _try_import_pandas()
    if pd is not None:
        if isinstance(data, pd.DataFrame):
            return _normalize_pandas_df(data)
        if isinstance(data, pd.Series):
            return data.tolist()

    torch = _try_import_torch()
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().tolist()

    nx = _try_import_networkx()
    if nx is not None and isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return _normalize_networkx_graph(data, nx)

    # ── Generators / Iterators ────────────────────────────────────
    if _is_generator(data):
        return _materialize_generator(data)

    # ── Generic iterable fallback (last resort) ──────────────────
    if _is_iterable(data):
        return _materialize_generator(iter(data))

    # ── Fallback ──────────────────────────────────────────────────
    return str(data)


def detect_type(data: Any) -> str:
    """Detect the type of data and return a descriptive string."""
    if data is None:
        return "none"
    if isinstance(data, bool):
        return "bool"
    if isinstance(data, int):
        return "int"
    if isinstance(data, float):
        return "float"
    if isinstance(data, str):
        return "str"
    if isinstance(data, (list, tuple)):
        return f"list({len(data)})"
    if isinstance(data, dict):
        return f"dict({len(data)} keys)"
    if isinstance(data, set):
        return f"set({len(data)})"
    if isinstance(data, frozenset):
        return f"frozenset({len(data)})"
    type_name = type(data).__name__
    module = type(data).__module__
    if module != "builtins":
        return f"{module}.{type_name}"
    return type_name


def normalize_with_metadata(data: Any) -> dict:
    """Normalize data and return both the result and original type info."""
    original_type = detect_type(data)
    normalized = normalize(data)
    return {"original_type": original_type, "data": normalized}


# ═══════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _normalize_string(data: str, allow_file_read: bool = False, allow_url_fetch: bool = False) -> Any:
    stripped = data.strip()
    if allow_file_read and _looks_like_file(stripped):
        return _normalize_file(stripped)
    if allow_url_fetch and stripped.startswith(("http://", "https://")):
        return _normalize_url(stripped)
    if stripped.startswith(("{", "[")):
        try:
            return normalize(json.loads(stripped))
        except (json.JSONDecodeError, ValueError):
            pass
    return data


def _looks_like_file(s: str) -> bool:
    if not s:
        return False
    if s.startswith(("~", "/", "./", "..\\")):
        return True
    return Path(s).suffix in (
        ".json", ".csv", ".tsv", ".txt", ".parquet", ".arrow",
        ".npy", ".npz", ".pt", ".pkl", ".h5", ".hdf5",
        ".xlsx", ".xls", ".xml", ".yaml", ".yml", ".toml",
    )


def _normalize_file(path_str: str) -> Any:
    path = Path(path_str).expanduser()
    if not path.exists():
        return path_str
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return normalize(json.load(f))
    if suffix in (".csv", ".tsv"):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
            return {"columns": reader.fieldnames or [], "rows": rows}
    if suffix == ".txt":
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        with open(path, encoding="utf-8") as f:
            return normalize(json.load(f))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return path_str


def _normalize_url(url: str) -> Any:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            content = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return normalize(json.loads(content))
            if "csv" in ct:
                text = content.decode("utf-8")
                reader = csv.DictReader(io.StringIO(text))
                return {"columns": reader.fieldnames or [], "rows": list(reader)}
            return content.decode("utf-8", errors="replace")
    except Exception:
        return url


def _normalize_bytes(data: bytes) -> Any:
    try:
        return normalize(json.loads(data.decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        pass
    try:
        text = data.decode("utf-8")
        if "," in text.split("\n", 1)[0]:
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            result = {"columns": reader.fieldnames or [], "rows": rows}
            if rows:
                result["shape"] = [len(rows), len(reader.fieldnames or [])]
            if result["columns"]:
                return result
    except Exception:
        pass
    return base64.b64encode(data).decode("ascii")


def _normalize_numpy(data, np) -> Any:
    if data.ndim == 0:
        return data.item()
    return data.tolist()


def _normalize_pandas_df(data) -> Any:
    """Normalize a pandas DataFrame to {columns, rows, dtypes, shape}."""
    return {
        "columns": data.columns.tolist(),
        "rows": data.values.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "shape": list(data.shape),
    }


def _normalize_networkx_graph(data, nx) -> Any:
    """Normalize a NetworkX graph to {nodes, edges, directed, ...}."""
    nodes = [
        {"id": str(n), **({str(k): normalize(v) for k, v in d.items()} if d else {})}
        for n, d in data.nodes(data=True)
    ]
    edges = [
        {"source": str(u), "target": str(v), **({str(k): normalize(v) for k, v in d.items()} if d else {})}
        for u, v, d in data.edges(data=True)
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "directed": data.is_directed(),
        "num_nodes": data.number_of_nodes(),
        "num_edges": data.number_of_edges(),
    }


def _is_dataclass(obj) -> bool:
    return hasattr(type(obj), "__dataclass_fields__")


def _is_generator(obj) -> bool:
    if isinstance(obj, (str, bytes, list, tuple, dict, set, frozenset)):
        return False
    return hasattr(obj, "__next__")


def _materialize_generator(gen, cap: int = 10000) -> list:
    result = []
    try:
        for i, item in enumerate(gen):
            if i >= cap:
                break
            result.append(normalize(item))
    except StopIteration:
        pass
    return result


def _is_iterable(obj) -> bool:
    if isinstance(obj, (str, bytes, dict)):
        return False
    return hasattr(obj, "__iter__")


# ── Lazy import helpers ─────────────────────────────────────────

_numpy = None
_numpy_checked = False
def _try_import_numpy():
    global _numpy, _numpy_checked
    if not _numpy_checked:
        try:
            import numpy as m
            _numpy = m
        except ImportError:
            _numpy = None
        _numpy_checked = True
    return _numpy

_pandas = None
_pandas_checked = False
def _try_import_pandas():
    global _pandas, _pandas_checked
    if not _pandas_checked:
        try:
            import pandas as m
            _pandas = m
        except ImportError:
            _pandas = None
        _pandas_checked = True
    return _pandas

_torch = None
_torch_checked = False
def _try_import_torch():
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch as m
            _torch = m
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch

_nx = None
_nx_checked = False
def _try_import_networkx():
    global _nx, _nx_checked
    if not _nx_checked:
        try:
            import networkx as m
            _nx = m
        except ImportError:
            _nx = None
        _nx_checked = True
    return _nx
