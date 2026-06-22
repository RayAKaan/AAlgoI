from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class CognitiveAction(IntEnum):
    DECOMPOSE_PROBLEM = 0
    IDENTIFY_STRUCTURE = 1
    EXTRACT_CONSTRAINTS = 2
    FIND_INVARIANT = 3
    ESTIMATE_COMPLEXITY = 4

    QUERY_SIMILAR = 5
    QUERY_PRINCIPLE = 6
    QUERY_ALGORITHMS = 7
    QUERY_FAILURES = 8
    QUERY_COMPLEXITY = 9

    SELECT_ALGORITHM = 10
    MODIFY_ALGORITHM = 11
    COMBINE_ALGORITHMS = 12
    APPLY_OPTIMIZATION = 13
    SYNTHESIZE_NEW = 14
    DECOMPOSE_RECURSIVE = 15

    TEST_EXAMPLES = 16
    STRESS_TEST = 17
    CHECK_EDGE_CASES = 18
    VERIFY_COMPLEXITY = 19
    PROVE_CORRECTNESS = 20

    BACKTRACK = 21
    ACCEPT_SOLUTION = 22
    RECORD_DISCOVERY = 23
    REQUEST_SIMPLIFY = 24


@dataclass
class ActionParams:
    action: CognitiveAction
    algorithm_id: str | None = None
    modification_type: str | None = None
    optimization_type: str | None = None
    principle_name: str | None = None
    complexity_target: str | None = None
    extra: dict | None = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


@dataclass
class ActionResult:
    action: CognitiveAction
    success: bool
    output: Any
    solution_code: str | None
    correctness: float
    time_complexity: str | None
    space_complexity: str | None
    is_novel_algorithm: bool = False
    reward_signal: float = 0.0
    error: str | None = None
