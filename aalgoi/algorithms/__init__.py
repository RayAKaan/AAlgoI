import importlib

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import AlgorithmRegistry, algorithm, get_registry

_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    for mod_name in [
        "aalgoi.algorithms.sorting",
        "aalgoi.algorithms.searching",
        "aalgoi.algorithms.math",
        "aalgoi.algorithms.strings",
        "aalgoi.algorithms.graph",
        "aalgoi.algorithms.dp",
        "aalgoi.algorithms.optimization",
        "aalgoi.algorithms.ml",
        "aalgoi.algorithms.nlp",
        "aalgoi.algorithms.image",
    ]:
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass


# Trigger registration on import
_ensure_registered()


__all__ = [
    "Algorithm",
    "AlgorithmRegistry",
    "algorithm",
    "get_registry",
]
