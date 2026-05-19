import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from core.problem_spec import ProblemSpec, ProblemType
from core.problem_library import ProblemLibrary
from algorithms.primitives import (
    PRIMITIVES, get_primitive_names, get_composable_chain,
    compose_pipeline, Primitive
)

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    algorithms: List[Dict[str, Any]]
    strategy: str
    confidence: float
    explanation: str
    execution_time_ms: float = 0.0
    sources: List[str] = field(default_factory=list)
    pipeline: Optional[List[Primitive]] = None


class AlgorithmSynthesizer:
    def __init__(self, problem_library: Optional[ProblemLibrary] = None,
                 llm_client: Optional[Any] = None):
        self.problem_library = problem_library or ProblemLibrary()
        self.llm_client = llm_client
        self._synthesis_cache: Dict[str, SynthesisResult] = {}

    def synthesize(self, problem_spec: ProblemSpec,
                   use_llm: bool = False,
                   timeout_ms: int = 5000) -> SynthesisResult:
        start = time.perf_counter()
        cache_key = problem_spec.get_signature()

        if cache_key in self._synthesis_cache:
            cached = self._synthesis_cache[cache_key]
            cached.execution_time_ms = (time.perf_counter() - start) * 1000
            return cached

        result = self._cascade_synthesis(problem_spec, use_llm)

        result.execution_time_ms = (time.perf_counter() - start) * 1000
        self._synthesis_cache[cache_key] = result
        return result

    def _cascade_synthesis(self, spec: ProblemSpec, use_llm: bool) -> SynthesisResult:
        found, template_result = self._template_match(spec)
        if found:
            return template_result

        if self.problem_library and len(self.problem_library.problems) > 0:
            found, transfer_result = self._transfer_synthesis(spec)
            if found:
                return transfer_result

        if use_llm and self.llm_client:
            llm_result = self._llm_synthesis(spec)
            return llm_result

        composition_result = self._primitive_composition(spec)
        return composition_result

    def _template_match(self, spec: ProblemSpec) -> Tuple[bool, SynthesisResult]:
        algorithms = []
        pt = spec.problem_type

        if pt == ProblemType.OPTIMIZATION or pt == ProblemType.ROUTING:
            has_minimize = any(o.direction == "minimize" for o in spec.objectives)
            if spec.constraints:
                algorithms.append({
                    "name": "greedy",
                    "type": "template",
                    "confidence": 0.6,
                    "complexity": "O(n log n)"
                })
            algorithms.append({
                "name": "dynamic_programming",
                "type": "template",
                "confidence": 0.5,
                "complexity": "O(n²)"
            })

        elif pt == ProblemType.SEARCH:
            has_sorted = any("sorted" in str(c).lower() for c in spec.constraints)
            if has_sorted:
                algorithms.append({
                    "name": "binary_search",
                    "type": "template",
                    "confidence": 0.8,
                    "complexity": "O(log n)"
                })
            algorithms.append({
                "name": "linear_search",
                "type": "template",
                "confidence": 0.7,
                "complexity": "O(n)"
            })

        elif pt == ProblemType.TRANSFORMATION:
            algorithms.append({
                "name": "quicksort",
                "type": "template",
                "confidence": 0.7,
                "complexity": "O(n log n)"
            })
            algorithms.append({
                "name": "mergesort",
                "type": "template",
                "confidence": 0.7,
                "complexity": "O(n log n)"
            })

        elif pt == ProblemType.CLASSIFICATION:
            algorithms.append({
                "name": "greedy",
                "type": "template",
                "confidence": 0.5,
                "complexity": "O(n log n)"
            })

        elif pt == ProblemType.SCHEDULING:
            algorithms.append({
                "name": "topological_sort",
                "type": "template",
                "confidence": 0.6,
                "complexity": "O(V + E)"
            })
            algorithms.append({
                "name": "greedy",
                "type": "template",
                "confidence": 0.6,
                "complexity": "O(n log n)"
            })

        if not algorithms:
            return False, None

        return True, SynthesisResult(
            algorithms=algorithms,
            strategy="template_match",
            confidence=max(a["confidence"] for a in algorithms),
            explanation=f"Matched {pt.value} problem type with {len(algorithms)} template algorithm(s)",
            sources=["built-in templates"]
        )

    def _transfer_synthesis(self, spec: ProblemSpec) -> Tuple[bool, SynthesisResult]:
        if not self.problem_library or not self.problem_library.problems:
            return False, None

        similar = self.problem_library.find_similar(spec, top_k=3, min_similarity=0.3)
        if not similar:
            return False, None

        best_algos = self.problem_library.get_best_algorithms(spec, top_k=5)
        if not best_algos:
            return False, None

        algorithms = []
        sources = []
        max_conf = 0.0

        for algo_name, score in best_algos:
            if algo_name in PRIMITIVES:
                algorithms.append({
                    "name": algo_name,
                    "type": "transfer",
                    "confidence": score * 0.9,
                    "complexity": getattr(PRIMITIVES[algo_name], 'time_complexity', 'O(n)')
                })
                max_conf = max(max_conf, score * 0.9)

        for result in similar:
            sources.append(result.get("name", "unknown"))

        if not algorithms:
            return False, None

        return True, SynthesisResult(
            algorithms=algorithms,
            strategy="transfer",
            confidence=max_conf,
            explanation=f"Transferred from {len(sources)} similar problem(s): {', '.join(sources[:3])}",
            sources=sources
        )

    def _llm_synthesis(self, spec: ProblemSpec) -> SynthesisResult:
        prompt = self._build_llm_prompt(spec)
        code = ""
        explanation = ""

        if self.llm_client:
            try:
                response = self.llm_client.generate(prompt)
                code = response.get("code", "")
                explanation = response.get("explanation", "")
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")

        return SynthesisResult(
            algorithms=[{
                "name": "llm_generated",
                "type": "llm",
                "confidence": 0.4,
                "complexity": "unknown",
                "code": code
            }],
            strategy="llm",
            confidence=0.4,
            explanation=explanation or "Generated via LLM (low confidence, please validate)"
        )

    def _build_llm_prompt(self, spec: ProblemSpec) -> str:
        return json.dumps({
            "task": "Generate an algorithm for the given problem specification",
            "problem": spec.to_dict(),
            "available_primitives": list(PRIMITIVES.keys()),
            "output_format": {
                "code": "python function implementation",
                "explanation": "step-by-step explanation"
            }
        })

    def _primitive_composition(self, spec: ProblemSpec) -> SynthesisResult:
        pipeline = self._infer_pipeline(spec)
        algorithms = []

        if pipeline:
            pipeline_info = []
            for p in pipeline:
                pipeline_info.append({
                    "name": p.name,
                    "complexity": p.time_complexity
                })
            algorithms.append({
                "name": "composed_pipeline",
                "type": "composition",
                "confidence": 0.5,
                "pipeline": [p.name for p in pipeline],
                "complexities": [p.time_complexity for p in pipeline]
            })

        fallbacks = self._get_fallback_algorithms(spec)
        algorithms.extend(fallbacks)

        conf = max((a.get("confidence", 0) for a in algorithms), default=0.5)

        return SynthesisResult(
            algorithms=algorithms,
            strategy="primitive_composition",
            confidence=conf,
            explanation=f"Composed {len(algorithms)} algorithm candidate(s) from primitives",
            pipeline=pipeline
        )

    def _infer_pipeline(self, spec: ProblemSpec) -> Optional[List[Primitive]]:
        pt = spec.problem_type
        has_constraints = len(spec.constraints) > 0
        has_numeric_output = any(
            "int" in str(v.get("type", "")) or "float" in str(v.get("type", ""))
            for v in spec.outputs.values()
        )
        has_list_output = any(
            "list" in str(v.get("type", ""))
            for v in spec.outputs.values()
        )

        if pt == ProblemType.SEARCH:
            search_target_name = "search" if "search" in spec.name.lower() else None
            chain = get_composable_chain("iterate", "binary_search")
            if chain:
                return chain

        elif pt == ProblemType.OPTIMIZATION:
            if has_constraints:
                chain = get_composable_chain("iterate", "reduce")
                if chain:
                    return chain

        elif pt == ProblemType.TRANSFORMATION:
            if has_list_output:
                return compose_pipeline(["iterate", "map"])

        return None

    def _get_fallback_algorithms(self, spec: ProblemSpec) -> List[Dict]:
        algorithms = []

        input_types = set()
        for inp in spec.inputs.values():
            t = str(inp.get("type", "")).lower()
            if "list" in t:
                input_types.add("list")
            elif "int" in t or "float" in t:
                input_types.add("numeric")
            elif "str" in t:
                input_types.add("string")

        if "list" in input_types:
            algorithms.append({
                "name": "iterate",
                "type": "fallback",
                "confidence": 0.3,
                "complexity": "O(n)"
            })
            algorithms.append({
                "name": "map",
                "type": "fallback",
                "confidence": 0.3,
                "complexity": "O(n)"
            })
            if spec.problem_type == ProblemType.SEARCH:
                algorithms.append({
                    "name": "linear_search",
                    "type": "fallback",
                    "confidence": 0.4,
                    "complexity": "O(n)"
                })

        if "string" in input_types and spec.problem_type == ProblemType.SEARCH:
            algorithms.append({
                "name": "rabin_karp",
                "type": "fallback",
                "confidence": 0.3,
                "complexity": "O(n+m)"
            })

        return algorithms

    def clear_cache(self):
        self._synthesis_cache.clear()
