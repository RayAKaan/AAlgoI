import re
from typing import Dict, List, Optional, Tuple
from core.problem_spec import ProblemSpec, ProblemType, Objective, Constraint

PATTERNS: List[Dict] = [
    {
        "name": "sorting",
        "keywords": ["sort", "order", "arrange", "sorted", "alphabetical", "ascending", "descending"],
        "problem_type": ProblemType.TRANSFORMATION,
        "output_type": "list",
        "objectives": [],
        "constraints": []
    },
    {
        "name": "search",
        "keywords": ["find", "search", "locate", "lookup", "contains", "index of"],
        "problem_type": ProblemType.SEARCH,
        "output_type": "int",
        "objectives": [],
        "constraints": []
    },
    {
        "name": "optimization",
        "keywords": ["maximize", "minimize", "optimize", "best", "optimal", "optimal"],
        "problem_type": ProblemType.OPTIMIZATION,
        "output_type": "float",
        "objectives": [],
        "constraints": []
    },
    {
        "name": "routing",
        "keywords": ["route", "path", "shortest path", "travel", "tour", "distance", "map"],
        "problem_type": ProblemType.ROUTING,
        "output_type": "list",
        "objectives": [],
        "constraints": ["shortest distance"]
    },
    {
        "name": "scheduling",
        "keywords": ["schedule", "deadline", "time slot", "assign", "resource"],
        "problem_type": ProblemType.SCHEDULING,
        "output_type": "list",
        "objectives": [],
        "constraints": []
    },
    {
        "name": "classification",
        "keywords": ["classify", "category", "label", "type of", "group"],
        "problem_type": ProblemType.CLASSIFICATION,
        "output_type": "str",
        "objectives": [],
        "constraints": []
    },
    {
        "name": "clustering",
        "keywords": ["cluster", "group similar", "segment"],
        "problem_type": ProblemType.CLUSTERING,
        "output_type": "list",
        "objectives": [],
        "constraints": []
    },
]


def parse_description(description: str) -> ProblemSpec:
    text = description.lower()

    detected_type = ProblemType.TRANSFORMATION
    objectives = []
    constraints = []
    name = description[:60]

    for pattern in PATTERNS:
        matches = sum(1 for kw in pattern["keywords"] if kw in text)
        if matches >= 2:
            detected_type = pattern["problem_type"]
            objectives = [
                Objective(desc) if isinstance(desc, str) else desc
                for desc in pattern.get("objectives", [])
            ]
            constraints = pattern.get("constraints", [])
            break
        elif matches == 1 and detected_type == ProblemType.TRANSFORMATION:
            detected_type = pattern["problem_type"]
            constraints = pattern.get("constraints", [])

    direction = _detect_direction(text)
    metric = _detect_metric(text)
    if direction and metric:
        objectives = [Objective(direction, metric)]

    inputs = _infer_inputs(text)
    outputs = _infer_outputs(text, detected_type)

    return ProblemSpec(
        name=name,
        problem_type=detected_type,
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        objectives=objectives,
        description=description
    )


def _detect_direction(text: str) -> Optional[str]:
    if any(w in text for w in ["maximize", "maximum", "max", "largest", "fastest", "biggest"]):
        return "maximize"
    if any(w in text for w in ["minimize", "minimum", "min", "smallest", "shortest", "cheapest"]):
        return "minimize"
    return None


def _detect_metric(text: str) -> Optional[str]:
    metrics = ["time", "distance", "cost", "profit", "accuracy", "speed",
               "weight", "length", "area", "volume", "score", "efficiency"]
    for m in metrics:
        if m in text:
            return m
    return None


def _infer_inputs(text: str) -> Dict:
    inputs = {}
    if any(w in text for w in ["list", "array", "numbers", "items", "data"]):
        inputs["data"] = {"type": "list"}
    if any(w in text for w in ["graph", "network", "nodes", "edges"]):
        inputs["graph"] = {"type": "dict"}
    if "string" in text or "text" in text:
        inputs["text"] = {"type": "str"}
    if not inputs:
        inputs["data"] = {"type": "list"}
    return inputs


def _infer_outputs(text: str, ptype: ProblemType) -> Dict:
    if ptype in (ProblemType.SEARCH, ProblemType.CLASSIFICATION):
        return {"result": {"type": "int"}}
    if ptype in (ProblemType.OPTIMIZATION,):
        return {"best": {"type": "float"}}
    return {"result": {"type": "list"}}


def extract_data_from_description(description: str) -> Optional[List]:
    numbers = re.findall(r'\d+', description)
    if numbers:
        return [int(n) for n in numbers]
    return None


def parse_solve_input(user_input: str) -> Tuple[ProblemSpec, Optional[List]]:
    data = extract_data_from_description(user_input)
    spec = parse_description(user_input)
    return spec, data
