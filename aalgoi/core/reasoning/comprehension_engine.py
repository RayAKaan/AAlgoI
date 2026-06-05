from typing import Any
import re

from aalgoi.core.reasoning.essence import ProblemEssence


class DeepComprehensionEngine:
    STRUCTURE_KEYWORDS = {
        "optimal_substructure": [
            "longest", "shortest", "minimum cost", "maximum profit",
            "number of ways", "count paths", "subsequence",
            "edit distance", "minimum operations",
        ],
        "greedy_exchange": [
            "interval", "schedule", "earliest", "deadline",
            "huffman", "minimum spanning", "activity",
        ],
        "monotonic_feasibility": [
            "minimum maximum", "maximum minimum",
            "at least k", "capacity", "feasible",
            "split array", "koko eating",
        ],
        "amortized_invariant": [
            "next greater", "next smaller", "daily temperatures",
            "sliding window maximum", "largest rectangle",
        ],
        "graph_connectivity": [
            "path", "connected", "reach", "route",
            "shortest path", "bfs", "dijkstra",
        ],
        "total_order": [
            "sort", "order", "ascending", "descending",
            "rank", "arrange",
        ],
        "hashing_fingerprint": [
            "anagram", "duplicate", "frequency",
            "substring", "pattern match", "two sum",
        ],
        "divide_conquer": [
            "merge sort", "quick select", "majority element",
            "closest pair", "binary search",
        ],
    }

    def comprehend(
        self,
        problem_text: str,
        data: Any = None,
    ) -> ProblemEssence:
        text_lower = problem_text.lower()
        hidden_structure = self._detect_structure(text_lower)
        domain = self._detect_domain(text_lower, data)
        goal = self._detect_goal(text_lower)
        constraints = self._parse_constraints(problem_text)
        n = constraints.get("n", 10**5)
        time_budget = ProblemEssence._derive_time_budget(n)
        output_structure = self._detect_output_structure(data, goal)

        return ProblemEssence(
            domain=domain,
            input_structure=self._detect_input_structure(data),
            output_structure=output_structure,
            optimization_goal=goal,
            constraints=constraints,
            time_budget=time_budget,
            hidden_structure=hidden_structure,
        )

    def _detect_structure(self, text: str) -> str:
        best_structure = "unknown"
        best_count = 0
        for structure, keywords in self.STRUCTURE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_structure = structure
        return best_structure

    def _detect_domain(self, text: str, data: Any) -> str:
        if data is not None:
            if isinstance(data, dict):
                if "edges" in data or "graph" in str(data).lower():
                    return "graph"
                if "nums" in data or "arr" in data:
                    return "integers"
                if "text" in data or "s" in data:
                    return "text"
            if isinstance(data, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in data[:10]):
                    return "integers"
                if all(isinstance(x, str) for x in data[:10]):
                    return "text"
        if any(w in text for w in ["graph", "edge", "node", "vertex", "path"]):
            return "graph"
        if any(w in text for w in ["sort", "order", "arrange"]):
            return "integers"
        if any(w in text for w in ["string", "text", "word", "character"]):
            return "text"
        return "unknown"

    def _detect_goal(self, text: str) -> str:
        goal_keywords = {
            "minimize": ["minimum", "min", "smallest", "shortest", "least"],
            "maximize": ["maximum", "max", "largest", "longest", "greatest"],
            "count": ["count", "number of", "how many"],
            "check": ["check", "determine", "whether", "is it"],
            "find": ["find", "get", "return", "compute", "calculate"],
        }
        for goal, keywords in goal_keywords.items():
            if any(kw in text for kw in keywords):
                return goal
        return "find"

    def _parse_constraints(self, text: str) -> dict:
        constraints = {}
        n_match = re.search(r'(?:n|length)\s*<=\s*10\^?(\d+)', text)
        if n_match:
            constraints["n"] = 10 ** int(n_match.group(1))
        n_direct = re.search(r'(?:n|length)\s*<=\s*(\d+)', text)
        if n_direct and "n" not in constraints:
            constraints["n"] = int(n_direct.group(1))
        return constraints

    def _detect_input_structure(self, data: Any) -> str:
        if data is None:
            return "unknown"
        if isinstance(data, (list, tuple)):
            return f"array of {len(data)} elements"
        if isinstance(data, dict):
            keys = list(data.keys())
            return f"dict with keys {keys[:5]}"
        return "unknown"

    def _detect_output_structure(self, data: Any, goal: str) -> str:
        if goal in ("count", "minimize", "maximize"):
            return "integer"
        if goal == "check":
            return "boolean"
        if goal == "find":
            return "array"
        return "unknown"
