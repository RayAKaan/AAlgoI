from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SecurityPolicy:
    allow_imports: bool = False
    allow_eval: bool = False
    allow_file_io: bool = False
    allow_network: bool = False
    max_execution_time_s: float = 5.0
    max_memory_mb: float = 256.0
    allowed_builtins: set[str] = field(default_factory=lambda: {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "int", "isinstance", "len", "list", "map", "max",
        "min", "range", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip",
    })


DefaultPolicy = SecurityPolicy()
