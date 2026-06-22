
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ProblemType(Enum):
    SORTING = "sorting"
    OPTIMIZATION = "optimization"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEARCH = "search"
    TRANSFORMATION = "transformation"
    GENERATION = "generation"
    DECISION = "decision"
    CLUSTERING = "clustering"
    SCHEDULING = "scheduling"
    ROUTING = "routing"
    PATHFINDING = "pathfinding"
    ML = "ml"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    IMAGE_PROCESSING = "image_processing"
    UNKNOWN = "unknown"


@dataclass
class Objective:
    direction: str  # "minimize" or "maximize"
    metric: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(f"Invalid direction: {self.direction}")


@dataclass
class Constraint:
    description: str
    validator: Callable | None = None

    def __str__(self) -> str:
        return self.description


@dataclass
class ProblemSpec:
    name: str
    problem_type: ProblemType = ProblemType.OPTIMIZATION
    inputs: dict[str, dict] = field(default_factory=dict)
    outputs: dict[str, dict] = field(default_factory=dict)
    constraints: list[Any] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    validation: dict[str, Callable] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    domain_knowledge: dict = field(default_factory=dict)
    description: str = ""

    sub_problems: dict[str, 'ProblemSpec'] = field(default_factory=dict)
    pipeline_order: list[str] = field(default_factory=list)

    def is_multi_domain(self) -> bool:
        return len(self.sub_problems) > 0 or self._has_heterogeneous_inputs()

    def _has_heterogeneous_inputs(self) -> bool:
        if len(self.inputs) > 3:
            types = [v.get('type', '') for v in self.inputs.values()]
            return len(set(types)) > 1
        return False

    def decompose(self, data: dict) -> list[tuple['ProblemSpec', Any]]:
        if not self.pipeline_order:
            self.pipeline_order = self._infer_pipeline_order()
        decomposed = []
        for domain in self.pipeline_order:
            if domain in self.sub_problems:
                sub_spec = self.sub_problems[domain]
                sub_data = data.get(domain, {})
                decomposed.append((sub_spec, sub_data))
            else:
                decomposed.append((self, data))
        return decomposed

    def _infer_pipeline_order(self) -> list[str]:
        return list(self.sub_problems.keys())

    def get_time_budget(self) -> float:
        return next(
            (float(c.description.split('=')[1]) for c in self.constraints
             if 'time_budget_ms' in str(c)),
            1000.0
        ) / 1000.0

    def get_priority(self) -> str:
        return next(
            (c.description.split('=')[1].strip().lower() for c in self.constraints
             if 'priority' in str(c)),
            'balanced'
        )

    CONSTRAINT_KEYWORDS = {
        "unique": 0, "sorted": 1, "capacity": 2,
        "time": 3, "distance": 4, "optimal": 5,
        "budget": 6, "weight": 7, "order": 8,
        "overlap": 9, "balance": 10, "coverage": 11,
        "priority": 12, "deadline": 13, "path": 14
    }

    INFERRED_PATTERNS = {
        "routing": {
            "keywords": ["route", "path", "tour", "travel", "shortest", "distance", "visit"],
            "type": ProblemType.ROUTING
        },
        "scheduling": {
            "keywords": ["schedule", "shift", "time slot", "assign", "resource", "deadline"],
            "type": ProblemType.SCHEDULING
        },
        "classification": {
            "keywords": ["classify", "category", "label", "predict", "type of"],
            "type": ProblemType.CLASSIFICATION
        },
        "sorting": {
            "keywords": ["sort", "order", "arrange", "sorted"],
            "type": ProblemType.SORTING
        },
        "search": {
            "keywords": ["find", "search", "locate", "lookup"],
            "type": ProblemType.SEARCH
        },
        "pathfinding": {
            "keywords": ["shortest path", "route", "navigate", "path between", "graph path", "find path", "a*", "dijkstra"],
            "type": ProblemType.PATHFINDING
        },
        "optimization": {
            "keywords": ["optimize", "maximize", "minimize", "knapsack", "allocate", "resource", "best value", "simulated annealing"],
            "type": ProblemType.OPTIMIZATION
        },
        "ml": {
            "keywords": ["train", "model", "learn", "predict", "classify", "regress", "embedding", "neural"],
            "type": ProblemType.ML
        },
        "nlp": {
            "keywords": ["word", "text", "sentence", "embedding", "semantic", "language", "corpus", "word2vec"],
            "type": ProblemType.NLP
        },
        "computer_vision": {
            "keywords": ["detect", "segment", "recognize", "object", "image", "visual"],
            "type": ProblemType.COMPUTER_VISION
        },
        "image_processing": {
            "keywords": ["blur", "filter", "denoise", "edge", "enhance", "gaussian", "median"],
            "type": ProblemType.IMAGE_PROCESSING
        }
    }

    def __post_init__(self) -> None:
        self.constraints = [
            c if isinstance(c, Constraint) else Constraint(str(c))
            for c in self.constraints
        ]
        self.objectives = [
            o if isinstance(o, Objective) else Objective(**o) if isinstance(o, dict) else o
            for o in self.objectives
        ]

    def infer_problem_type(self, data: Any = None) -> ProblemType:
        if data is not None:
            detected = self._infer_from_data_shape(data)
            if detected != ProblemType.UNKNOWN:
                return detected

        combined = f"{self.name} {self.description} {' '.join(str(c) for c in self.constraints)}".lower()

        for pattern_name, info in self.INFERRED_PATTERNS.items():
            if any(kw in combined for kw in info["keywords"]):
                matches = sum(1 for kw in info["keywords"] if kw in combined)
                if matches >= 2:
                    return info["type"]

        if not self.objectives:
            if any(t in str(self.outputs) for t in ["class", "label"]):
                return ProblemType.CLASSIFICATION
            if any(t in str(self.outputs) for t in ["value", "number", "float"]):
                return ProblemType.REGRESSION
            if any(t in str(self.outputs) for t in ["list", "sorted", "transformed"]):
                return ProblemType.SORTING

        # Backward compat: TRANSFORMATION → SORTING
        if self.problem_type == ProblemType.TRANSFORMATION:
            return ProblemType.SORTING

        return self.problem_type

    @staticmethod
    def _infer_from_data_shape(data: Any) -> ProblemType:
        if isinstance(data, tuple) and len(data) == 3:
            if isinstance(data[0], dict):
                return ProblemType.PATHFINDING
        if isinstance(data, tuple) and len(data) == 2:
            if isinstance(data[0], list) and isinstance(data[1], (int, float)):
                return ProblemType.OPTIMIZATION
        if isinstance(data, dict):
            if "graph" in data:
                return ProblemType.PATHFINDING
            if "items" in data and "capacity" in data:
                return ProblemType.OPTIMIZATION
            if "corpus" in data or "model" in data:
                return ProblemType.ML
            if "embeddings" in data:
                return ProblemType.NLP
            if "X_train" in data:
                y = data.get("y_train")
                if y is not None:
                    if hasattr(y, 'shape'):
                        import numpy as np
                        unique = np.unique(y)
                        if len(unique) < 20:
                            return ProblemType.CLASSIFICATION
                        return ProblemType.REGRESSION
                    if isinstance(y, list):
                        unique = list(set(y))
                        if all(isinstance(v, str) for v in unique):
                            return ProblemType.CLASSIFICATION
                        if all(isinstance(v, (int, float)) for v in unique):
                            return ProblemType.REGRESSION
                return ProblemType.CLUSTERING
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
            return ProblemType.SORTING
        if hasattr(data, 'shape') and len(data.shape) == 2:
            return ProblemType.CLUSTERING
        return ProblemType.UNKNOWN

    def to_vector(self) -> np.ndarray:
        VEC_SIZE = 128
        vector = np.zeros(VEC_SIZE)

        all_types = sorted(pt.value for pt in ProblemType)
        type_idx = all_types.index(self.problem_type.value) if self.problem_type.value in all_types else 0
        if type_idx < VEC_SIZE:
            vector[type_idx] = 1.0

        input_count = len(self.inputs)
        vector[10] = min(input_count / 10, 1.0)
        type_set = set()
        for inp in self.inputs.values():
            t = str(inp.get("type", "")).lower()
            if "list" in t:
                type_set.add("list")
            elif "int" in t or "float" in t:
                type_set.add("numeric")
            elif "str" in t:
                type_set.add("string")
            elif "tuple" in t:
                type_set.add("tuple")
            elif "dict" in t:
                type_set.add("dict")
        vector[11] = len(type_set) / 5.0

        output_count = len(self.outputs)
        vector[20] = min(output_count / 10, 1.0)

        obj_dirs = [o.direction for o in self.objectives]
        vector[40] = obj_dirs.count("minimize") / max(len(obj_dirs), 1)
        vector[41] = obj_dirs.count("maximize") / max(len(obj_dirs), 1)

        weights = [o.weight for o in self.objectives]
        if weights:
            vector[42] = np.mean(weights)
            vector[43] = np.std(weights) if len(weights) > 1 else 0

        combined_text = f"{self.name} {' '.join(str(c) for c in self.constraints)} {self.description}".lower()
        words = re.findall(r'\w+', combined_text)
        word_counts = Counter(words)
        top_words = [w for w, _ in word_counts.most_common(50)]

        for i, word in enumerate(top_words[:50]):
            idx = 50 + i
            if idx < VEC_SIZE:
                vector[idx] = word_counts[word] / max(word_counts.most_common(1)[0][1], 1)

        for constraint in self.constraints:
            desc = str(constraint).lower()
            for keyword, idx in self.CONSTRAINT_KEYWORDS.items():
                if keyword in desc:
                    vec_idx = 30 + idx
                    if vec_idx < 40:
                        vector[vec_idx] = 1.0

        example_count = len(self.examples)
        vector[100] = min(example_count / 10, 1.0)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def validate(self) -> tuple[bool, list[str]]:
        errors = []

        if not self.name:
            errors.append("Problem name is required")

        if not self.inputs:
            errors.append("At least one input is required")

        if not self.outputs:
            errors.append("At least one output is required")

        if not self.objectives and self.problem_type in (ProblemType.OPTIMIZATION, ProblemType.ROUTING, ProblemType.SCHEDULING, ProblemType.SORTING):
            errors.append(f"Problem type {self.problem_type.value} needs at least one objective")

        if not self.examples:
            pass

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "problem_type": self.problem_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "constraints": [str(c) for c in self.constraints],
            "objectives": [{"direction": o.direction, "metric": o.metric, "weight": o.weight} for o in self.objectives],
            "examples": self.examples,
            "description": self.description,
            "sub_problems": {k: v.to_dict() for k, v in self.sub_problems.items()},
            "pipeline_order": self.pipeline_order
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSpec":
        raw_sub = d.get("sub_problems", {})
        sub_problems = {k: cls.from_dict(v) for k, v in raw_sub.items()} if isinstance(raw_sub, dict) else {}
        return cls(
            name=d.get("name", ""),
            problem_type=ProblemType(d.get("problem_type", "optimization")) if isinstance(d.get("problem_type"), str) else d.get("problem_type", ProblemType.OPTIMIZATION),
            inputs=d.get("inputs", {}),
            outputs=d.get("outputs", {}),
            constraints=[Constraint(**c) if isinstance(c, dict) else c for c in d.get("constraints", [])],
            objectives=[Objective(**o) if isinstance(o, dict) else o for o in d.get("objectives", [])],
            examples=d.get("examples", []),
            description=d.get("description", ""),
            domain_knowledge=d.get("domain_knowledge", {}),
            sub_problems=sub_problems,
            pipeline_order=d.get("pipeline_order", [])
        )

    def get_signature(self) -> str:
        inp_sig = ",".join(sorted(f"{n}:{v.get('type','?')}" for n, v in self.inputs.items()))
        out_sig = ",".join(sorted(f"{n}:{v.get('type','?')}" for n, v in self.outputs.items()))
        obj_sig = ",".join(f"{o.direction[0]}{o.metric}" for o in self.objectives)
        return f"{self.problem_type.value}|{inp_sig}|{out_sig}|{obj_sig}"
