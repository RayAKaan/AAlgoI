from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProblemTask(Enum):
    SORT = "sort"
    COUNTING_SORT = "counting_sort"
    LINEAR_SEARCH = "linear_search"
    BINARY_SEARCH = "binary_search"
    TWO_SUM = "two_sum"
    LOWER_BOUND = "lower_bound"
    GCD = "gcd"
    LCM = "lcm"
    IS_PRIME = "is_prime"
    SIEVE = "sieve"
    FAST_EXPONENTIATION = "fast_exponentiation"
    FIBONACCI = "fibonacci"
    PALINDROME = "palindrome"
    ANAGRAM = "anagram"
    KMP = "kmp"
    RABIN_KARP = "rabin_karp"
    EDIT_DISTANCE = "edit_distance"
    LCS = "lcs"
    BFS = "bfs"
    DFS = "dfs"
    SHORTEST_PATH_UNWEIGHTED = "shortest_path_unweighted"
    SHORTEST_PATH_WEIGHTED = "shortest_path_weighted"
    SHORTEST_PATH_NEGATIVE = "shortest_path_negative"
    TOPOLOGICAL_SORT = "topological_sort"
    CYCLE_DETECTION = "cycle_detection"
    CONNECTED_COMPONENTS = "connected_components"
    MST = "mst"
    MAX_FLOW = "max_flow"
    KADANE = "kadane"
    KNAPSACK_01 = "knapsack_01"
    KNAPSACK_FRACTIONAL = "knapsack_fractional"
    COIN_CHANGE = "coin_change"
    LIS = "lis"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_SUMMARIZATION = "text_summarization"
    IMAGE_BLUR = "image_blur"
    EDGE_DETECTION = "edge_detection"


class Domain(Enum):
    SORTING = "sorting"
    SEARCHING = "searching"
    MATH = "math"
    STRINGS = "strings"
    GRAPH = "graph"
    DP = "dp"
    OPTIMIZATION = "optimization"
    ML = "ml"
    NLP = "nlp"
    IMAGE = "image"


DOMAIN_FOR_TASK: dict[ProblemTask, Domain] = {
    ProblemTask.SORT: Domain.SORTING,
    ProblemTask.COUNTING_SORT: Domain.SORTING,
    ProblemTask.LINEAR_SEARCH: Domain.SEARCHING,
    ProblemTask.BINARY_SEARCH: Domain.SEARCHING,
    ProblemTask.TWO_SUM: Domain.SEARCHING,
    ProblemTask.LOWER_BOUND: Domain.SEARCHING,
    ProblemTask.GCD: Domain.MATH,
    ProblemTask.LCM: Domain.MATH,
    ProblemTask.IS_PRIME: Domain.MATH,
    ProblemTask.SIEVE: Domain.MATH,
    ProblemTask.FAST_EXPONENTIATION: Domain.MATH,
    ProblemTask.FIBONACCI: Domain.MATH,
    ProblemTask.PALINDROME: Domain.STRINGS,
    ProblemTask.ANAGRAM: Domain.STRINGS,
    ProblemTask.KMP: Domain.STRINGS,
    ProblemTask.RABIN_KARP: Domain.STRINGS,
    ProblemTask.EDIT_DISTANCE: Domain.STRINGS,
    ProblemTask.LCS: Domain.STRINGS,
    ProblemTask.BFS: Domain.GRAPH,
    ProblemTask.DFS: Domain.GRAPH,
    ProblemTask.SHORTEST_PATH_UNWEIGHTED: Domain.GRAPH,
    ProblemTask.SHORTEST_PATH_WEIGHTED: Domain.GRAPH,
    ProblemTask.SHORTEST_PATH_NEGATIVE: Domain.GRAPH,
    ProblemTask.TOPOLOGICAL_SORT: Domain.GRAPH,
    ProblemTask.CYCLE_DETECTION: Domain.GRAPH,
    ProblemTask.CONNECTED_COMPONENTS: Domain.GRAPH,
    ProblemTask.MST: Domain.GRAPH,
    ProblemTask.MAX_FLOW: Domain.GRAPH,
    ProblemTask.KADANE: Domain.DP,
    ProblemTask.KNAPSACK_01: Domain.DP,
    ProblemTask.KNAPSACK_FRACTIONAL: Domain.OPTIMIZATION,
    ProblemTask.COIN_CHANGE: Domain.DP,
    ProblemTask.LIS: Domain.DP,
    ProblemTask.CLASSIFICATION: Domain.ML,
    ProblemTask.REGRESSION: Domain.ML,
    ProblemTask.CLUSTERING: Domain.ML,
    ProblemTask.DIMENSIONALITY_REDUCTION: Domain.ML,
    ProblemTask.ANOMALY_DETECTION: Domain.ML,
    ProblemTask.SENTIMENT_ANALYSIS: Domain.NLP,
    ProblemTask.TEXT_SUMMARIZATION: Domain.NLP,
    ProblemTask.IMAGE_BLUR: Domain.IMAGE,
    ProblemTask.EDGE_DETECTION: Domain.IMAGE,
}


class SolveMode(Enum):
    DETERMINISTIC = "deterministic"
    LEARNED = "learned"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class Constraints:
    time_budget_ms: float = 1000.0
    memory_mb: float = 256.0
    input_size_limit: int | None = None
    allow_approximate: bool = False
    allowed_domains: tuple[Domain, ...] = ()
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Objective:
    direction: str = "minimize"
    metric: str = "time"
    weight: float = 1.0


@dataclass(frozen=True)
class Example:
    input: Any
    output: Any
    description: str = ""


@dataclass(frozen=True)
class Precondition:
    name: str
    check: Callable[[Any], bool]
    description: str = ""


@dataclass(frozen=True)
class Complexity:
    time: str
    space: str
    time_raw: str = ""
    space_raw: str = ""


@dataclass(frozen=True)
class ProblemSpec:
    id: str
    task: ProblemTask
    domain: Domain
    inputs: dict[str, Any]
    constraints: Constraints = field(default_factory=Constraints)
    objective: Objective = field(default_factory=Objective)
    expected_output_type: str = "Any"
    examples: tuple[Example, ...] = ()
    confidence: float = 0.0
    parse_trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    task: ProblemTask
    domain: Domain
    preconditions: tuple[Precondition, ...] = ()
    complexity: Complexity = field(default_factory=lambda: Complexity("?", "?"))
    principles: frozenset[str] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)
    deterministic: bool = True
    exact: bool = True
    validator: Callable[[Any, Any], bool] | None = None
    generator: Callable[[], tuple[Any, Any]] | None = None
    examples: tuple[Example, ...] = ()


@dataclass
class ValidationResult:
    passed: bool = False
    oracle_match: bool = False
    property_match: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    algorithm: str
    task: ProblemTask
    score: float
    source: str = ""
    reason: str = ""


@dataclass
class DecisionEvent:
    step: str
    detail: str
    time_ms: float = 0.0


@dataclass
class SolveResult:
    output: Any = None
    ok: bool = False
    algorithm: str | None = None
    validated: bool = False
    validation: ValidationResult = field(default_factory=ValidationResult)
    candidates: list[CandidateScore] = field(default_factory=list)
    complexity: Complexity | None = None
    time_ms: float = 0.0
    mode: str = "deterministic"
    confidence: float = 0.0
    error: str | None = None
    trace: list[DecisionEvent] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.ok


@dataclass
class BenchmarkReport:
    suite: str = ""
    total: int = 0
    correct: int = 0
    failed: int = 0
    errors: int = 0
    accuracy: float = 0.0
    by_task: dict[str, dict] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    time_ms: float = 0.0
