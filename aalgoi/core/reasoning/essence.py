from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemEssence:
    domain: str = "unknown"
    input_structure: str = "unknown"
    output_structure: str = "unknown"
    invariant: str = ""
    optimization_goal: str = "find"
    optimization_metric: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    time_budget: str = "unknown"
    math_objects: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    hidden_structure: str = "unknown"

    @classmethod
    def from_problem_spec(cls, spec: "ProblemSpec") -> "ProblemEssence":
        domain_map = {
            "SORTING": "integers",
            "PATHFINDING": "graph",
            "OPTIMIZATION": "numbers",
            "ML_CLASSIFICATION": "feature_matrix",
            "ML_REGRESSION": "feature_matrix",
            "CLUSTERING": "feature_matrix",
            "NLP": "text",
            "SEARCH": "array",
            "STRING": "string",
            "IMAGE": "image",
        }
        goal_map = {
            "SORTING": "arrange",
            "PATHFINDING": "find",
            "OPTIMIZATION": "minimize",
            "ML_CLASSIFICATION": "classify",
            "ML_REGRESSION": "predict",
            "CLUSTERING": "partition",
            "NLP": "transform",
            "SEARCH": "find",
        }
        structure_map = {
            "SORTING": "total_order",
            "PATHFINDING": "graph_connectivity",
            "OPTIMIZATION": "optimal_substructure",
            "ML_CLASSIFICATION": "statistical_separation",
            "ML_REGRESSION": "statistical_fitting",
            "SEARCH": "ordered_or_unordered",
        }

        domain = domain_map.get(spec.problem_type, "unknown")
        goal = goal_map.get(spec.problem_type, "find")
        structure = structure_map.get(spec.problem_type, "unknown")

        constraints = {}
        if spec.constraints:
            constraints = dict(spec.constraints)
        if spec.data_size:
            constraints["n"] = spec.data_size

        time_budget = cls._derive_time_budget(constraints.get("n", 10**5))

        return cls(
            domain=domain,
            input_structure=f"array of {domain}" if domain != "unknown" else "unknown",
            output_structure="array" if domain == "integers" else "unknown",
            optimization_goal=goal,
            constraints=constraints,
            time_budget=time_budget,
            hidden_structure=structure,
        )

    @staticmethod
    def _derive_time_budget(n: int) -> str:
        if n <= 20:
            return "O(2^n)"
        if n <= 1000:
            return "O(n^2)"
        if n <= 10**5:
            return "O(n log n)"
        if n <= 10**6:
            return "O(n)"
        return "O(log n)"
