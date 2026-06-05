from typing import TYPE_CHECKING, Any

from aalgoi.core.mind.synthesis.modifier import CodeModifier
from aalgoi.core.mind.synthesis.optimizer import CodeOptimizer
from aalgoi.core.mind.synthesis.templates import TemplateRegistry

if TYPE_CHECKING:
    from aalgoi.core.mind.mind_state import MindState


class SynthesizerAdapter:
    def __init__(self, synthesizer: Any = None) -> None:
        self._external_synthesizer = synthesizer
        self.templates = TemplateRegistry()
        self.modifier = CodeModifier()
        self.optimizer = CodeOptimizer()

    def synthesize_novel(
        self,
        problem_text: str,
        data: Any,
        principle: str | None = None,
        complexity_target: str | None = None,
        failed_approaches: list[str] | None = None,
        kg_context: Any = None,
    ) -> str | None:
        problem_info = self._extract_problem_info(problem_text, data)

        code = self.templates.find_best(principle, problem_info)
        if code:
            return code

        code = self.templates.find_best(None, problem_info)
        if code:
            return code

        if self._external_synthesizer:
            try:
                result = self._external_synthesizer.synthesize(
                    problem_text=problem_text,
                    data=data,
                    principle=principle,
                )
                if result and result.code:
                    return result.code
            except Exception:
                pass

        return None

    def modify(
        self,
        base_algorithm_code: str,
        problem_text: str,
        state: "MindState | None" = None,
        modification_type: str | None = None,
    ) -> str | None:
        if not base_algorithm_code:
            return None

        problem_info = self._extract_problem_info(problem_text, None)

        if modification_type:
            return self.modifier.modify(
                base_algorithm_code, modification_type, problem_info
            )

        modification_types = [
            "add_memoization",
            "sort_first",
            "optimize_inner_loop",
            "add_early_termination",
            "add_hashing",
        ]

        for mod_type in modification_types:
            result = self.modifier.modify(
                base_algorithm_code, mod_type, problem_info
            )
            if result:
                return result

        return None

    def combine(
        self,
        algo1: dict,
        algo2: dict,
        state: "MindState | None" = None,
    ) -> str | None:
        code1 = algo1.get("code") or algo1.get("algorithm")
        code2 = algo2.get("code") or algo2.get("algorithm")

        if not code1 or not code2:
            return None

        combined = f'''def solve(data):
    # Stage 1: {algo1.get("algorithm", "algorithm_1")}
    result1 = (lambda data: {self._extract_body(code1)})(data)
    # Stage 2: {algo2.get("algorithm", "algorithm_2")}
    result2 = (lambda data: {self._extract_body(code2)})(result1)
    return result2
'''
        try:
            compile(combined, "<combine>", "exec")
            return combined
        except SyntaxError:
            return None

    def apply_optimization(
        self,
        code: str,
        optimization_type: str,
        data: Any = None,
    ) -> str | None:
        if not code:
            return None

        return self.optimizer.optimize(code, optimization_type, data)

    def _extract_problem_info(
        self,
        problem_text: str,
        data: Any,
    ) -> dict:
        text_lower = problem_text.lower()

        info = {
            "domain": "unknown",
            "optimization_goal": "find",
            "metric": "",
            "contiguity": "contiguous",
            "has_target": False,
            "has_pairs": False,
            "has_intervals": False,
            "has_capacity": False,
            "has_search_space": False,
            "has_two_sequences": False,
            "items_are_discrete": True,
            "items_reusable": False,
            "can_sort": True,
            "needs_frequency": False,
            "needs_next_greater": False,
            "needs_shortest": False,
            "needs_range_query": False,
        }

        if any(w in text_lower for w in ["graph", "edge", "node", "vertex", "path"]):
            info["domain"] = "graph"
        elif any(w in text_lower for w in ["string", "text", "word", "char"]):
            info["domain"] = "text"
        elif any(w in text_lower for w in ["array", "nums", "list", "element"]):
            info["domain"] = "array"
        elif any(w in text_lower for w in ["number", "int", "float", "value"]):
            info["domain"] = "integers"

        if any(w in text_lower for w in ["maximum", "max", "largest", "longest"]):
            info["optimization_goal"] = "maximize"
        elif any(w in text_lower for w in ["minimum", "min", "smallest", "shortest"]):
            info["optimization_goal"] = "minimize"
        elif any(w in text_lower for w in ["count", "number of", "how many"]):
            info["optimization_goal"] = "count"
        elif any(w in text_lower for w in ["check", "whether", "is it", "determine if"]):
            info["optimization_goal"] = "check"

        if any(w in text_lower for w in ["sum", "total"]):
            info["metric"] = "sum"
        elif any(w in text_lower for w in ["length", "size", "longest", "shortest"]):
            info["metric"] = "length"
        elif any(w in text_lower for w in ["count", "number"]):
            info["metric"] = "count"
        elif any(w in text_lower for w in ["cost", "price", "expense"]):
            info["metric"] = "cost"
        elif any(w in text_lower for w in ["profit", "gain", "revenue"]):
            info["metric"] = "profit"

        info["has_target"] = "target" in text_lower or "k" in text_lower
        info["has_pairs"] = any(
            w in text_lower for w in ["pair", "two sum", "two number", "complement"]
        )
        info["has_intervals"] = any(
            w in text_lower for w in ["interval", "range", "start", "end"]
        )
        info["has_capacity"] = any(
            w in text_lower for w in ["capacity", "weight", "knapsack", "limit", "budget"]
        )
        info["has_search_space"] = any(
            w in text_lower for w in ["minimum maximum", "maximum minimum", "split array", "capacity"]
        )
        info["has_two_sequences"] = any(
            w in text_lower for w in ["two string", "s1", "s2", "edit distance", "common"]
        )
        info["items_reusable"] = any(
            w in text_lower for w in ["unlimited", "unbounded", "coin change", "reuse"]
        )
        info["needs_frequency"] = any(
            w in text_lower for w in ["frequency", "frequent", "count of", "appear", "majority"]
        )
        info["needs_next_greater"] = any(
            w in text_lower for w in ["next greater", "next larger", "daily temperature"]
        )
        info["needs_shortest"] = any(
            w in text_lower for w in ["shortest", "minimum step", "least"]
        )
        info["needs_range_query"] = any(
            w in text_lower for w in ["range sum", "subarray sum", "between", "query"]
        )

        if "subsequence" in text_lower:
            info["contiguity"] = "subsequence"
        elif "subarray" in text_lower or "contiguous" in text_lower:
            info["contiguity"] = "contiguous"
        elif "subset" in text_lower:
            info["contiguity"] = "subset"

        if data is not None:
            if isinstance(data, dict):
                if "nums" in data:
                    arr = data["nums"]
                    if isinstance(arr, list) and len(arr) > 0:
                        if all(isinstance(x, int) for x in arr[:5]):
                            info["domain"] = "integers"
                if "target" in data:
                    info["has_target"] = True
                if "edges" in data:
                    info["domain"] = "graph"

        return info

    def _extract_body(self, code: str) -> str:
        lines = code.split("\n")
        body_lines = []
        in_func = False
        for line in lines:
            if line.strip().startswith("def "):
                in_func = True
                continue
            if in_func:
                body_lines.append(line)
        return "\n".join(body_lines)
