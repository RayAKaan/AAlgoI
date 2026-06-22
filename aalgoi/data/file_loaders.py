from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_file(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    loaders = {
        ".json": _load_json,
        ".csv": _load_csv,
        ".txt": _load_text,
        ".npy": _load_npy,
    }
    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix}")
    return loader(p)


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_text(path: Path) -> str:
    with open(path) as f:
        return f.read()


def _load_npy(path: Path) -> Any:
    import numpy as np
    return np.load(path).tolist()
