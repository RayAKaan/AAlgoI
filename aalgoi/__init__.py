"""
AAlgoI — Adaptive Algorithm Intelligence
=========================================

Self-adaptive algorithm selection & universal problem solving.
Zero-cost import. Nothing heavy loads until first use.

Quick start:
    >>> from aalgoi import sort, path, knapsack
    >>> sort([3, 1, 4, 1, 5])
    [1, 1, 3, 4, 5]
    >>> path(graph, "A", "D")
    ['A', 'B', 'D']
    >>> knapsack(items, capacity=50)
    {'selected': [...], 'value': 220}
"""

__version__ = "1.4.0"

__all__ = [
    # Core API
    "solve", "solve_spec", "explain", "why", "benchmark",
    # Shortcuts
    "sort", "sort_by", "rank", "path", "all_paths", "distance",
    "knapsack", "minimize", "maximize", "cluster", "classify",
    "regress", "embed", "reduce", "blur", "denoise", "edges",
    "enhance", "search", "compare",
    # Data models
    "ProblemSpec", "ProblemType", "Objective", "Constraint",
    # Classes
    "SmartSolver", "UniversalSolver", "Result",
    "Explainer", "Explanation", "Benchmarker", "DynamicRegistry",
    "RewardShaper", "AlgorithmMarketplace", "AlgorithmMetadata",
    "SandboxRL",
    # Namespaces
    "transformer", "model", "mind",
]

# ── Solver singletons (lazy, cached) ──────────────────────────────────────────

_solver = None
_transformer_parser = None


def _get_solver():
    """Load SmartSolver on first call. Cache forever."""
    global _solver
    if _solver is None:
        from core.smart_solver import SmartSolver
        _solver = SmartSolver()
    return _solver


def _get_transformer():
    """Load DistilBERT parser on first call. Cache forever."""
    global _transformer_parser
    if _transformer_parser is None:
        from core.question_parser import QuestionParser
        _transformer_parser = QuestionParser(use_transformer=True)
    return _transformer_parser


# ── Public API ────────────────────────────────────────────────────────────────


def solve(question, data=None, **kwargs):
    """
    Solve an algorithmic problem using natural language.
    Loads the RL agent on first call (~2s). Subsequent calls ~5ms.

    Returns a Result object with .result, .algorithm, .time_ms, .ok, .answer.

    Examples:
        >>> r = solve("sort ascending", [3, 1, 4, 1, 5])
        >>> r.result
        [1, 1, 3, 4, 5]
        >>> r.ok
        True
    """
    cfg = kwargs.pop("config", None)
    d = _get_solver().ask(question, data, **kwargs)
    from pipeline import Result
    return Result(
        result=d.get("result"),
        algorithm=d.get("algorithm", ""),
        time_ms=d.get("time_ms", 0.0),
        success=d.get("success", False),
        answer=d.get("answer", ""),
    )


def solve_spec(spec, data=None):
    """
    Solve using an explicit ProblemSpec.
    Faster and deterministic for programmatic use.

    Returns a Result object with .result, .algorithm, .time_ms, .ok.

    Examples:
        >>> from aalgoi import ProblemSpec, ProblemType
        >>> spec = ProblemSpec("sort", problem_type=ProblemType.SORTING)
        >>> r = solve_spec(spec, [3, 1, 2])
        >>> r.result
        [1, 2, 3]
    """
    from pipeline import UniversalSolver, Result
    solver = UniversalSolver()
    d = solver.solve(spec, data)
    return Result(
        result=d.get("result"),
        algorithm=d.get("algorithm", ""),
        time_ms=d.get("time_ms", 0.0),
        success=d.get("success", False),
    )


def explain(result_or_algo, data=None):
    """
    Get a human-readable explanation for an algorithm choice.

    Parameters
    ----------
    result_or_algo : Result, dict, or str
        A result (from solve()) containing 'algorithm' key,
        or an algorithm name string
    data : any, optional
        Input data or detail string passed through to Explainer

    Returns
    -------
    Explanation
        Structured explanation object with .summary, .complexity, .steps

    Examples
    --------
        >>> explain("timsort").summary
        'Timsort is a hybrid stable sorting algorithm...'
    """
    from core.explainer import Explainer
    exp = Explainer()
    if isinstance(result_or_algo, (dict, object)):
        algo_name = (
            result_or_algo.get("algorithm", "unknown")
            if isinstance(result_or_algo, dict)
            else getattr(result_or_algo, "algorithm", "unknown")
            if not isinstance(result_or_algo, str)
            else result_or_algo
        )
        return exp.explain(algo_name, detail=str(data) if data else None)
    return exp.explain(result_or_algo, detail=str(data) if data else None)


def why(result) -> str:
    """
    Return a human-readable explanation of why an algorithm was chosen.

    Examples:
        >>> r = solve("sort", [3, 1, 2])
        >>> why(r)
        'Chose timsort because data is nearly sorted (O(n) best case).'
    """
    from core.explainer import Explainer
    exp = Explainer()
    algo = result.get("algorithm", "unknown") if isinstance(result, dict) else getattr(result, "algorithm", "unknown")
    return exp.explain(algo).summary


def benchmark(spec, data):
    """
    Compare aalgoi against Python stdlib on the same problem.

    Returns a dict with keys:
        aalgoi_time_ms, baseline_time_ms, speedup_factor,
        aalgoi_algorithm, baseline_algorithm, winner

    Examples:
        >>> from aalgoi import ProblemSpec, ProblemType
        >>> spec = ProblemSpec("bench", problem_type=ProblemType.SORTING)
        >>> bm = benchmark(spec, [3, 1, 4, 1, 5])
        >>> bm['winner']
        'Baseline'
    """
    from pipeline import UniversalSolver
    from core.benchmarker import Benchmarker
    return Benchmarker(solver=UniversalSolver()).compare(spec, data)


# ── Transformer Namespace ─────────────────────────────────────────────────────


class transformer:
    """
    aalgoi.transformer — DistilBERT-powered NL classification.
    Loads ~270MB model on first call. Use for ambiguous queries.

    Examples:
        aalgoi.transformer.solve("pack these items efficiently", items)
        aalgoi.mind.solve("find the quickest way from A to D", graph)
    """
    @staticmethod
    def solve(query: str, data=None):
        """
        Solve using transformer-based NL understanding.
        Better than solve() for ambiguous natural language.
        Loads DistilBERT on first call (~3s). ~800ms after.
        """
        spec = _get_transformer().parse(query, data)
        return _get_solver().ask_with_spec(spec, data)

    @staticmethod
    def parse(query: str, data=None):
        """Return the ProblemSpec without executing."""
        return _get_transformer().parse(query, data)


class model:
    """Alias for aalgoi.transformer."""
    solve = transformer.solve
    parse = transformer.parse


class mind:
    """Alias for aalgoi.transformer."""
    solve = transformer.solve
    parse = transformer.parse


# ── Lazy __getattr__ — handles `from aalgoi import X` ────────────────────────

_LAZY_MAP = {
    # Pipeline
    "UniversalSolver":       ("pipeline",                  "UniversalSolver"),
    "Result":                ("pipeline",                  "Result"),
    # Core
    "SmartSolver":           ("core.smart_solver",         "SmartSolver"),
    "ProblemSpec":           ("core.problem_spec",         "ProblemSpec"),
    "ProblemType":           ("core.problem_spec",         "ProblemType"),
    "Objective":             ("core.problem_spec",         "Objective"),
    "Constraint":            ("core.problem_spec",         "Constraint"),
    "Explainer":             ("core.explainer",            "Explainer"),
    "Explanation":           ("core.explainer",            "Explanation"),
    "Benchmarker":           ("core.benchmarker",          "Benchmarker"),
    "DynamicRegistry":       ("core.registry_manager",     "DynamicRegistry"),
    "RewardShaper":          ("core.rl.reward_shaper",     "RewardShaper"),
    "AlgorithmMarketplace":  ("core.algorithm_marketplace","AlgorithmMarketplace"),
    "AlgorithmMetadata":     ("core.algorithm_marketplace","AlgorithmMetadata"),
    # Sandbox (optional — requires torch)
    "SandboxRL":             ("aalgoi.sandbox",            "SandboxRL"),
    # Shortcuts
    "sort":                  ("aalgoi.shortcuts",          "sort"),
    "sort_by":               ("aalgoi.shortcuts",          "sort_by"),
    "rank":                  ("aalgoi.shortcuts",          "rank"),
    "path":                  ("aalgoi.shortcuts",          "path"),
    "all_paths":             ("aalgoi.shortcuts",          "all_paths"),
    "distance":              ("aalgoi.shortcuts",          "distance"),
    "knapsack":              ("aalgoi.shortcuts",          "knapsack"),
    "minimize":              ("aalgoi.shortcuts",          "minimize"),
    "maximize":              ("aalgoi.shortcuts",          "maximize"),
    "cluster":               ("aalgoi.shortcuts",          "cluster"),
    "classify":              ("aalgoi.shortcuts",          "classify"),
    "regress":               ("aalgoi.shortcuts",          "regress"),
    "embed":                 ("aalgoi.shortcuts",          "embed"),
    "reduce":                ("aalgoi.shortcuts",          "reduce"),
    "blur":                  ("aalgoi.shortcuts",          "blur"),
    "denoise":               ("aalgoi.shortcuts",          "denoise"),
    "edges":                 ("aalgoi.shortcuts",          "edges"),
    "enhance":              ("aalgoi.shortcuts",          "enhance"),
    "search":                ("aalgoi.shortcuts",          "search"),
    "compare":               ("aalgoi.shortcuts",          "compare"),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        import importlib
        mod_path, attr = _LAZY_MAP[name]
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr)
            globals()[name] = obj
            return obj
        except (AttributeError, ImportError) as e:
            if name == "SandboxRL":
                raise ImportError(
                    "SandboxRL requires PyTorch: pip install torch"
                ) from e
            raise AttributeError(
                f"module 'aalgoi' has no attribute '{name}'. "
                f"Available: {sorted(__all__)}"
            ) from e
    raise AttributeError(
        f"module 'aalgoi' has no attribute '{name}'. "
        f"Available: {sorted(__all__)}"
    )
