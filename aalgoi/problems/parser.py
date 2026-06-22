from __future__ import annotations

import re
from typing import Any

from aalgoi.types import (
    Constraints, Domain, Example, Objective, ProblemSpec, ProblemTask,
    DOMAIN_FOR_TASK,
)


class ProblemParser:
    def parse(self, problem_text: str, data: Any = None) -> ProblemSpec:
        trace: list[str] = []
        text = problem_text.lower().strip()

        task = self._infer_task(text, data)
        trace.append(f"inferred_task={task.value}")
        domain = DOMAIN_FOR_TASK.get(task, Domain.MATH)
        trace.append(f"domain={domain.value}")

        inputs = self._extract_inputs(text, data)
        trace.append(f"inputs_keys={list(inputs.keys())}")

        constraints = self._infer_constraints(text, data)
        objective = self._infer_objective(text, task)
        examples = self._infer_examples(data)

        spec_id = re.sub(r'[^a-z0-9_]', '_', text[:40].strip()) or "problem"

        return ProblemSpec(
            id=spec_id,
            task=task,
            domain=domain,
            inputs=inputs,
            constraints=constraints,
            objective=objective,
            expected_output_type=self._infer_output_type(task),
            examples=examples,
            confidence=self._compute_confidence(task, text, data),
            parse_trace=tuple(trace),
        )

    def _infer_task(self, text: str, data: Any) -> ProblemTask:
        rules = [
            # ML / data-science tasks before broad generic fallbacks.
            (r'\b(?:classify|classification|classifier|predict.?class|label)\b', ProblemTask.CLASSIFICATION),
            (r'\b(?:regress|regression|predict.?value|forecast|estimate)\b', ProblemTask.REGRESSION),
            (r'\b(?:cluster|clustering|kmeans|k-means|dbscan|agglomerative|gaussian.?mixture)\b', ProblemTask.CLUSTERING),
            (r'\b(?:pca|dimensionality.?reduction|reduce.?dimensions|components)\b', ProblemTask.DIMENSIONALITY_REDUCTION),
            (r'\b(?:anomaly|outlier|isolation.?forest)\b', ProblemTask.ANOMALY_DETECTION),
            (r'\b(?:sentiment|emotion|polarity)\b', ProblemTask.SENTIMENT_ANALYSIS),
            (r'\b(?:summari[sz]e|summary|summarization)\b', ProblemTask.TEXT_SUMMARIZATION),
            (r'\b(?:gaussian.?blur|blur image|image blur)\b', ProblemTask.IMAGE_BLUR),
            (r'\b(?:edge.?detect|edge detection|sobel)\b', ProblemTask.EDGE_DETECTION),
            # Specific phrases must come before broad data/graph fallbacks.
            (r'\b(?:max(?:imum)?.?flow|edmonds.?karp)\b', ProblemTask.MAX_FLOW),
            (r'\bfind.?target\b|\bfind.?index\b|\bsearch.?target\b', ProblemTask.BINARY_SEARCH if 'sorted' in text else ProblemTask.LINEAR_SEARCH),
            (r'\balphabeti[sz]e\b', ProblemTask.SORT),
            (r'\bcounting.?sort\b', ProblemTask.COUNTING_SORT),
            (r'\bsort\b', ProblemTask.SORT),
            (r'\bbinary.?search\b', ProblemTask.BINARY_SEARCH),
            (r'\btwo.?sum\b', ProblemTask.TWO_SUM),
            (r'\blower.?bound\b', ProblemTask.LOWER_BOUND),
            (r'\blinear.?search\b', ProblemTask.LINEAR_SEARCH),
            (r'\bgcd\b', ProblemTask.GCD),
            (r'\blcm\b', ProblemTask.LCM),
            (r'\bprime\b', ProblemTask.IS_PRIME),
            (r'\bsieve\b', ProblemTask.SIEVE),
            (r'\bfast.?exponentiation\b', ProblemTask.FAST_EXPONENTIATION),
            (r'\bfib(?:onacci)?\b', ProblemTask.FIBONACCI),
            (r'\bpalindrome\b', ProblemTask.PALINDROME),
            (r'\banagram\b', ProblemTask.ANAGRAM),
            (r'\b(?:kmp|knuth.?morris.?pratt)\b', ProblemTask.KMP),
            (r'\brabin.?karp\b', ProblemTask.RABIN_KARP),
            (r'\bedit.?distance|levenshtein\b', ProblemTask.EDIT_DISTANCE),
            (r'\blongest.?common.?subsequence\b', ProblemTask.LCS),
            (r'\bbfs|breadth.?first\b', ProblemTask.BFS),
            (r'\bdfs|depth.?first\b', ProblemTask.DFS),
            (r'\bnegative.?weight|bellman.?ford\b', ProblemTask.SHORTEST_PATH_NEGATIVE),
            (r'\bunweighted\b', ProblemTask.SHORTEST_PATH_UNWEIGHTED),
            (r'\bshortest.?path\b', ProblemTask.SHORTEST_PATH_WEIGHTED),
            (r'\btopological|dependency.?order\b', ProblemTask.TOPOLOGICAL_SORT),
            (r'\bcycle.?detect|has.?cycle\b', ProblemTask.CYCLE_DETECTION),
            (r'\bconnected.?component(?:s)?\b', ProblemTask.CONNECTED_COMPONENTS),
            (r'\bmst|minimum.?spanning|kruskal\b', ProblemTask.MST),
            (r'\bkadane|maximum.?subarray\b', ProblemTask.KADANE),
            (r'\bknapsack\b', ProblemTask.KNAPSACK_01),
            (r'\bcoin.?change\b', ProblemTask.COIN_CHANGE),
            (r'\blis|longest.?increasing.?subsequence\b', ProblemTask.LIS),
        ]
        for pattern, task in rules:
            if re.search(pattern, text):
                return task

        if data is not None:
            inferred = self._infer_from_data(data)
            if inferred is not None:
                return inferred

        return ProblemTask.SORT

    def _infer_from_data(self, data: Any) -> ProblemTask | None:
        if isinstance(data, dict):
            keys = set(data)
            if {"X_train", "y_train"}.issubset(keys) or {"train_x", "train_y"}.issubset(keys):
                y = data.get("y_train", data.get("train_y", []))
                return ProblemTask.REGRESSION if _looks_regression_target(y) else ProblemTask.CLASSIFICATION
            if "n_clusters" in data or "k" in data:
                return ProblemTask.CLUSTERING
            if "n_components" in data or "components" in data:
                return ProblemTask.DIMENSIONALITY_REDUCTION
            if "contamination" in data or "outliers" in data:
                return ProblemTask.ANOMALY_DETECTION
            if "image" in data and "sigma" in data:
                return ProblemTask.IMAGE_BLUR
            if "image" in data and any(k in data for k in ["edges", "edge", "sobel"]):
                return ProblemTask.EDGE_DETECTION
            if "text" in data and "pattern" in data:
                return ProblemTask.KMP
            if "text" in data:
                return ProblemTask.SENTIMENT_ANALYSIS
            if "items" in data or "capacity" in data:
                return ProblemTask.KNAPSACK_01
            if ("nums" in data or "data" in data or "arr" in data) and "target" in data:
                return ProblemTask.LINEAR_SEARCH
            if "graph" in data or any(k in data for k in ["start", "end", "source"]):
                return ProblemTask.SHORTEST_PATH_UNWEIGHTED
            if "a" in data and "b" in data:
                vals = [v for v in data.values() if isinstance(v, int)]
                if len(vals) == 2:
                    return ProblemTask.GCD
        if isinstance(data, list):
            return ProblemTask.SORT
        return None

    def _extract_inputs(self, text: str, data: Any) -> dict[str, Any]:
        if data is not None:
            if isinstance(data, dict):
                return dict(data)
            if isinstance(data, list):
                return {"data": data}
            return {"value": data}
        nums = re.findall(r'-?\d+', text)
        if nums:
            return {"data": [int(n) for n in nums[:10]]}
        return {"data": []}

    def _infer_constraints(self, text: str, data: Any) -> Constraints:
        return Constraints()

    def _infer_objective(self, text: str, task: ProblemTask) -> Objective:
        return Objective()

    def _infer_examples(self, data: Any) -> tuple[Example, ...]:
        return ()

    def _infer_output_type(self, task: ProblemTask) -> str:
        return "Any"

    def _compute_confidence(self, task: ProblemTask, text: str, data: Any) -> float:
        return 0.8


def _looks_regression_target(y: Any) -> bool:
    try:
        vals = list(y)
    except Exception:
        return False
    if not vals:
        return False
    if any(isinstance(v, float) and not float(v).is_integer() for v in vals):
        return True
    return all(isinstance(v, (int, float)) for v in vals) and len(set(vals)) > max(10, len(vals) // 2)


parse_problem = ProblemParser().parse
