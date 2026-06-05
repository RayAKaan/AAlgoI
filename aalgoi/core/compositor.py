
import ast
import copy
import functools
import inspect
import textwrap
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aalgoi.core.pipeline_graph import PipelineGraph


class DynamicCompositor:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._modification_history: list[dict] = []
        self._cache_registry: dict[str, Any] = {}
        self._ast_cache: dict[str, ast.AST] = {}
        self._use_dag = self.config.get("enable_dag", True)

    def build_pipeline(self, algorithms: list[Any], context: dict[str, Any]) -> list[Any]:
        if self._use_dag and len(algorithms) > 1:
            return self._build_dag_pipeline(algorithms, context)
        return self._build_linear_pipeline(algorithms, context)

    def _build_linear_pipeline(self, algorithms: list[Any], context: dict[str, Any]) -> list[Any]:
        pipeline = []

        for i, algo in enumerate(algorithms):
            adapted = algo.clone() if hasattr(algo, 'clone') else copy.deepcopy(algo)
            adapted = self._adapt(adapted, context, pipeline_position=i)
            pipeline.append(adapted)

        return pipeline

    def _build_dag_pipeline(self, algorithms: list[Any], context: dict[str, Any]) -> list[Any]:
        graph = PipelineGraph()
        features = context.get("features", {})

        cpu_free = features.get("cpu_free", 0.5)
        parallel_ok = cpu_free > 0.5 and len(algorithms) > 1

        for i, algo in enumerate(algorithms):
            adapted = algo.clone() if hasattr(algo, 'clone') else copy.deepcopy(algo)
            adapted = self._adapt(adapted, context, pipeline_position=i)

            if parallel_ok and i > 0 and self._can_parallelize(adapted, context):
                graph.add_algorithm(f"step_{i}", adapted,
                                    depends_on=[f"step_{j}" for j in range(i) if not self._can_parallelize(
                                        algorithms[j], context)])
            else:
                deps = [f"step_{i-1}"] if i > 0 else []
                graph.add_algorithm(f"step_{i}", adapted, depends_on=deps)

        return graph

    def _can_parallelize(self, algo: Any, context: dict) -> bool:
        if any(tag in getattr(algo, 'tags', []) for tag in ["sorting", "stable"]):
            return False
        if hasattr(algo, 'name') and algo.name in ['timsort', 'merge_sort', 'heap_sort']:
            return False
        return True

    def _adapt(self, algo: Any, context: dict, pipeline_position: int = 0) -> Any:
        context.get("features", {})
        context.get("predictions", {})
        context.get("constraints", {})

        algo = self._parametric_tune(algo, context)

        if self._should_cache(algo, context):
            algo = self._inject_memoization(algo)

        if self._should_parallelize(algo, context):
            algo = self._inject_parallelization(algo, context)

        if pipeline_position > 0:
            algo = self._optimize_intermediate(algo, context)

        if self._should_ast_optimize(algo, context):
            algo = self._ast_optimize(algo, context)

        return algo

    def _parametric_tune(self, algo: Any, context: dict) -> Any:
        features = context.get("features", {})
        constraints = context.get("constraints", {})

        params = {}

        cpu_free = features.get("cpu_free", 0.5)
        if cpu_free > 0.7:
            params["threads"] = min(int(cpu_free * 8), 8)
            params["quality"] = "high"
        elif cpu_free > 0.4:
            params["threads"] = min(int(cpu_free * 4), 4)
            params["quality"] = "balanced"
        else:
            params["threads"] = 1
            params["quality"] = "fast"

        mem_free = features.get("mem_free_ratio", 0.5)
        if mem_free < 0.2:
            params["in_place"] = True
            params["buffer_size"] = "small"
        elif mem_free > 0.6:
            params["buffer_size"] = "large"

        time_budget = constraints.get("time_budget_ms", 500)
        if time_budget < 100:
            params["early_exit"] = True
            params["max_iterations"] = 100

        priority = constraints.get("priority", "balanced")
        if priority == "speed":
            params["approximate"] = True
            params["precision"] = "low"
        elif priority == "accuracy":
            params["approximate"] = False
            params["precision"] = "high"

        if hasattr(algo, "set_params"):
            algo.set_params(**params)

        return algo

    def _should_cache(self, algo: Any, context: dict) -> bool:
        data_profile = context.get("data_profile", {})
        patterns = data_profile.get("patterns", {})

        if patterns.get("has_repeated_subsequences", False):
            return True
        if hasattr(algo, "call_count") and algo.call_count > 5:
            return True
        if data_profile.get("size", 0) < 1000:
            return True

        return False

    def _inject_memoization(self, algo: Any) -> Any:
        if not hasattr(algo, "process"):
            return algo

        original_process = algo.process
        cache_size = self.config.get("cache_size", 256)

        @functools.lru_cache(maxsize=cache_size)
        def cached_process(data_tuple):
            data = list(data_tuple) if isinstance(data_tuple, tuple) else data_tuple
            return original_process(data)

        def wrapper(data):
            if isinstance(data, list):
                data_tuple = tuple(data)
            else:
                data_tuple = data

            try:
                return cached_process(data_tuple)
            except (TypeError, HashError):
                return original_process(data)

        algo.process = wrapper
        algo.modifications.append("memoization")

        return algo

    def _should_parallelize(self, algo: Any, context: dict) -> bool:
        if any(tag in algo.tags for tag in ["sorting", "stable"]):
            return False
        if hasattr(algo, 'name') and algo.name in ['timsort', 'merge_sort', 'heap_sort']:
            return False

        features = context.get("features", {})
        data_profile = context.get("data_profile", {})

        data_size = data_profile.get("size", 0)
        cpu_free = features.get("cpu_free", 0)
        cpu_count = features.get("cpu_count", 0)

        if data_size > 50000 and cpu_free > 0.5 and cpu_count > 1:
            return True

        predictions = context.get("predictions", {})
        if predictions.get("recommended_parallelism", 1) > 1:
            return True

        return False

    def _inject_parallelization(self, algo: Any, context: dict) -> Any:
        if not hasattr(algo, "process"):
            return algo

        original_process = algo.process
        features = context.get("features", {})

        cpu_count = max(1, int(features.get("cpu_count", 1) * 16))
        parallelism = min(cpu_count - 1, 8)

        def parallel_process(data):
            if not isinstance(data, list) or len(data) < 1000:
                return original_process(data)

            chunk_size = max(1, len(data) // parallelism)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                results = list(executor.map(original_process, chunks))

            merged = []
            for r in results:
                if isinstance(r, list):
                    merged.extend(r)
                else:
                    merged.append(r)

            return merged

        algo.process = parallel_process
        algo.modifications.append(f"parallelization({parallelism})")

        return algo

    def _optimize_intermediate(self, algo: Any, context: dict) -> Any:
        constraints = context.get("constraints", {})

        if constraints.get("priority") == "speed":
            if hasattr(algo, "set_params"):
                algo.set_params(streaming=True, buffer_size="small")

        return algo

    def _should_ast_optimize(self, algo: Any, context: dict) -> bool:
        if hasattr(algo, "call_count") and algo.call_count > 20:
            return True
        predictions = context.get("predictions", {})
        if predictions.get("confidence", 0) > 0.8:
            return True
        return False

    def _ast_optimize(self, algo: Any, context: dict) -> Any:
        try:
            if not hasattr(algo, "process"):
                return algo

            source = inspect.getsource(algo.process)
            source = textwrap.dedent(source)

            tree = ast.parse(source)

            optimizer = ASTOptimizer(context)
            tree = optimizer.visit(tree)
            ast.fix_missing_locations(tree)

            code = compile(tree, "<optimized>", "exec")
            namespace = {}
            exec(code, namespace)

            for name, obj in namespace.items():
                if callable(obj) and name != "__builtins__":
                    algo.process = types.MethodType(obj, algo)
                    algo.modifications.append("ast_optimization")
                    break

        except (OSError, TypeError, SyntaxError):
            pass

        return algo

    def execute_dag(self, graph: PipelineGraph, data: Any) -> Any:
        return graph.execute(data)

    def get_modification_stats(self) -> dict[str, Any]:
        return {
            "total_modifications": len(self._modification_history),
            "modification_types": list(set(
                m["type"] for m in self._modification_history
            ))
        }


class ASTOptimizer(ast.NodeTransformer):
    def __init__(self, context: dict):
        self.context = context
        self.optimizations_applied = []

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)

        if (len(node.body) == 1 and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Call)):

            call = node.body[0].value
            if (isinstance(call.func, ast.Attribute) and
                call.func.attr == "append"):
                self.optimizations_applied.append("list_comprehension")

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)

        has_cache = any(
            isinstance(d, ast.Call) and
            isinstance(d.func, ast.Attribute) and
            d.func.attr == "lru_cache"
            for d in node.decorator_list
        )

        if not has_cache:
            cache_decorator = ast.parse("@functools.lru_cache(maxsize=128)").body[0].decorator_list[0]
            node.decorator_list.insert(0, cache_decorator)
            self.optimizations_applied.append("auto_cache")

        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        return node


class HashError(Exception):
    pass
