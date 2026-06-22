from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, name: str | None = None) -> str:
        if name is None:
            name = f"ckpt_{int(time.time())}"
        meta = {"name": name, "created_at": time.time(), "version": 3}
        ckpt_path = self.path / f"{name}.json"
        with open(ckpt_path, "w") as f:
            json.dump(meta, f)
        return str(ckpt_path)

    def restore(self, target: str = "last_good") -> dict:
        ckpts = sorted(self.path.glob("*.json"))
        if not ckpts:
            return {"success": False, "error": "no_checkpoints"}
        ckpt = ckpts[-1]
        with open(ckpt) as f:
            meta = json.load(f)
        return {"success": True, "target": meta.get("name", str(ckpt.stem))}

    def list_checkpoints(self) -> list[dict]:
        results = []
        for ckpt in sorted(self.path.glob("*.json")):
            with open(ckpt) as f:
                meta = json.load(f)
            results.append({"name": ckpt.stem, "created_at": meta.get("created_at", 0), "version": meta.get("version", 0)})
        return results
