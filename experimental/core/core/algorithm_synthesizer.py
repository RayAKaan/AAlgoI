import ast
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.primitives import (
    PRIMITIVES,
    Primitive,
    compose_pipeline,
    get_composable_chain,
)
from aalgoi.core.ast_optimizer import ASTOptimizer
from aalgoi.core.llm_client import OllamaClient
from aalgoi.core.problem_library import ProblemLibrary
from aalgoi.core.problem_spec import ProblemSpec, ProblemType
from aalgoi.core.prompt_builder import build_synthesis_prompt
from aalgoi.core.sandboxed_executor import (
    benchmark_sandboxed,
    create_sandboxed_module,
    execute_sandboxed,
)

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    algorithms: list[dict[str, Any]]
    strategy: str
    confidence: float
    explanation: str
    execution_time_ms: float = 0.0
    sources: list[str] = field(default_factory=list)
    pipeline: list[Primitive] | None = None


class AlgorithmSynthesizer:
    def __init__(self, problem_library: ProblemLibrary | None = None,
                 llm_client: Any | None = None):
        self.problem_library = problem_library or ProblemLibrary()
        self.llm_client = llm_client
        self._synthesis_cache: dict[str, SynthesisResult] = {}

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

    def _template_match(self, spec: ProblemSpec) -> tuple[bool, SynthesisResult]:
        algorithms = []
        pt = spec.problem_type

        if pt == ProblemType.OPTIMIZATION or pt == ProblemType.ROUTING:
            any(o.direction == "minimize" for o in spec.objectives)
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

    def _transfer_synthesis(self, spec: ProblemSpec) -> tuple[bool, SynthesisResult]:
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

    def _infer_pipeline(self, spec: ProblemSpec) -> list[Primitive] | None:
        pt = spec.problem_type
        has_constraints = len(spec.constraints) > 0
        any(
            "int" in str(v.get("type", "")) or "float" in str(v.get("type", ""))
            for v in spec.outputs.values()
        )
        has_list_output = any(
            "list" in str(v.get("type", ""))
            for v in spec.outputs.values()
        )

        if pt == ProblemType.SEARCH:
            "search" if "search" in spec.name.lower() else None
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

    def _get_fallback_algorithms(self, spec: ProblemSpec) -> list[dict]:
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

    def clear_cache(self) -> None:
        self._synthesis_cache.clear()


class LLMAlgorithmSynthesizer:
    def __init__(self, ollama_url: str = None, model: str = "llama2"):
        self.llm = OllamaClient(base_url=ollama_url, model=model)

    def synthesize(
        self,
        spec: ProblemSpec,
        data: Any,
        baseline_algo: Algorithm | None = None,
        min_improvement: float = 0.05,
    ) -> Algorithm | None:
        if not self._should_attempt(spec, data):
            return None

        prompt = build_synthesis_prompt(spec, data[:10] if isinstance(data, list) else data)
        try:
            code = self.llm.generate(prompt, temperature=0.3, max_tokens=1000)
        except RuntimeError:
            return None
        if not code or "def process" not in code:
            return None

        optimizer = ASTOptimizer()
        code = optimizer.optimize(code)

        module = create_sandboxed_module("synthesized_algo", code)
        if not module:
            return None

        sample = data[:5] if isinstance(data, list) else data
        success, result = execute_sandboxed(module, "process", sample)
        if not success:
            return None

        if baseline_algo:
            ok, synth_time, base_time = benchmark_sandboxed(
                module, "process", data,
                lambda d: baseline_algo.process(d),
                trials=3,
            )
            if not ok:
                return None
            if synth_time > base_time * (1 - min_improvement):
                return None

        return self._module_to_algorithm(module, spec, code)

    def _should_attempt(self, spec: ProblemSpec, data: Any) -> bool:
        size = len(data) if hasattr(data, '__len__') else 0
        return 10 <= size <= 1000

    def _module_to_algorithm(
        self, module: Any, spec: ProblemSpec, original_code: str,
    ) -> Algorithm:
        tree = ast.parse(original_code)
        complexity = self._infer_complexity(tree)
        name_suffix = str(hash(original_code))[-8:]

        class SynthesizedAlgorithm(Algorithm):
            def __init__(self) -> None:
                super().__init__()
                self.name = f"synth_{spec.problem_type.name.lower()}_{name_suffix}"
                self.time_complexity = complexity.get('time', 'O(n)')
                self.space_complexity = complexity.get('space', 'O(n)')
                self.tags = [spec.problem_type.name.lower(), 'synthesized']
                self.best_for = [spec.problem_type.name.lower()]
                self._process_func = getattr(module, 'process')

            def process(self, data: Any) -> Any:
                return self._process_func(data)

        return SynthesizedAlgorithm()

    @staticmethod
    def _infer_complexity(tree: ast.Module) -> dict:
        loops = 0
        nested_loops = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loops += 1

        for node1 in ast.walk(tree):
            if isinstance(node1, (ast.For, ast.While)):
                for node2 in ast.walk(node1):
                    if node2 is not node1 and isinstance(node2, (ast.For, ast.While)):
                        nested_loops += 1

        if nested_loops > 0:
            time_complexity = "O(n^2)"
        elif loops > 0:
            time_complexity = "O(n)"
        else:
            time_complexity = "O(1)"

        return {
            'time': time_complexity,
            'space': "O(n)" if loops > 0 else "O(1)",
        }
