from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CodeTemplate:
    name: str
    principle: str
    description: str
    applicability: Callable[[dict], bool]
    generate: Callable[[dict], str]
