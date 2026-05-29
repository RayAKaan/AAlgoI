"""
aalgoi.sandbox.instance_store — Module-level named instance registry.

Supports create, get, list, delete, save-all, load-all.
Thread-safe via a module-level lock.
"""

from __future__ import annotations
import threading
import os
import json
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sandbox_rl import SandboxRL

_lock: threading.Lock = threading.Lock()
_instances: Dict[str, "SandboxRL"] = {}


def register(name: str, instance: "SandboxRL"):
    with _lock:
        _instances[name] = instance


def get_instance(name: str) -> Optional["SandboxRL"]:
    with _lock:
        return _instances.get(name)


def list_instances() -> List[str]:
    with _lock:
        return list(_instances.keys())


def delete_instance(name: str) -> bool:
    with _lock:
        if name in _instances:
            del _instances[name]
            return True
        return False


def save_all(directory: str):
    os.makedirs(directory, exist_ok=True)
    with _lock:
        index = {}
        for name, inst in _instances.items():
            safe_name = name.replace("/", "_").replace("\\", "_")
            path = os.path.join(directory, f"{safe_name}.rl")
            inst.save(path)
            index[name] = path
        with open(os.path.join(directory, "index.json"), "w") as f:
            json.dump(index, f, indent=2)


def load_all(directory: str):
    import importlib
    mod       = importlib.import_module(".sandbox_rl", package="aalgoi.sandbox")
    SandboxRL = getattr(mod, "SandboxRL")
    index_path = os.path.join(directory, "index.json")
    if not os.path.exists(index_path):
        return
    with open(index_path) as f:
        index = json.load(f)
    for name, path in index.items():
        if os.path.exists(path):
            inst = SandboxRL.from_checkpoint(path, name=name)
            register(name, inst)
