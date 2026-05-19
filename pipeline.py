
import time
import json
import math
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from core.context_engine import ContextEngine
from core.meta_controller import MetaController
from core.compositor import DynamicCompositor
from core.performance_tracker import PerformanceTracker
from core.knowledge_base import KnowledgeBase
from core.bandit import UCB1Bandit
from core.validator import PipelineValidator
from core.drift_detector import DriftDetector
from core.decision_log import DecisionLog, Decision
from core.pipeline_graph import PipelineGraph
from core.genetic_evolver import GeneticPipelineEvolver

from algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder
from algorithms.optimization import GreedyKnapsack, SimulatedAnnealing
from algorithms.safety import IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack

from core.problem_spec import ProblemSpec, ProblemType
from core.meta_controller import UniversalMetaController
from core.validator import LearningValidator
from core.explainer import Explainer
from algorithms.primitives import PRIMITIVES


@dataclass
class Result:
    output: Any = None
    algorithm: str = ""
    time_ms: float = 0.0
    success: bool = True
    metrics: Dict = field(default_factory=dict)
    pipeline: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "result": self.output,
            "algorithm": self.algorithm,
            "success": self.success,
            "time_ms": self.time_ms,
        }


class UniversalSolver:
    def __init__(self, problem_library=None, llm_client=None, config=None):
        self.config = config or {}
        self.registry = self._build_registry()
        self.meta_controller = UniversalMetaController(
            config=self.config,
            knowledge_base=None,
            algorithm_registry=self.registry,
        )
        self.context_engine = ContextEngine(config=self.config)
        self.validator = LearningValidator(
            adaptation_rate=self.config.get("adaptation_rate", 0.05)
        )
        self.explainer = Explainer(
            llm_client=llm_client,
            default_detail=self.config.get("explanation_detail", "short")
        )
        self._execution_count = 0

    def solve(self, problem_spec: ProblemSpec, data: Any,
              use_llm: bool = False,
              expected: Any = None) -> Dict:
        if problem_spec.is_multi_domain():
            return self._solve_multi_domain(problem_spec, data)
        return self._solve_single_domain(problem_spec, data, use_llm, expected)

    def _is_heterogeneous_dict(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if len(data) < 3:
            return False
        value_types = [type(v).__name__ for v in data.values()]
        if len(set(value_types)) <= 1:
            return False
        known_keys = {'packages', 'items', 'cities', 'graph', 'capacities', 'weights'}
        matching = [k for k in data if k in known_keys]
        return len(matching) >= 2

    def _solve_multi_domain(self, problem: ProblemSpec, data: Dict) -> Dict[str, Any]:
        if not problem.sub_problems:
            problem = self._auto_decompose_problem(problem, data)

        results = {}
        total_time = 0
        sub_algorithms = []
        all_ok = True

        for domain, (sub_spec, sub_data) in zip(problem.pipeline_order, problem.decompose(data)):
            logger.info("Solving sub-problem: %s", domain)
            sub_result = self._solve_single_domain(sub_spec, sub_data)
            results[domain] = sub_result['result']
            sub_algorithms.append(f"{domain}:{sub_result['algorithm']}")
            total_time += sub_result.get('time_ms', 0)
            if not sub_result.get('success', False):
                all_ok = False

        return {
            'result': results,
            'algorithm': f"pipeline:{' -> '.join(sub_algorithms)}",
            'success': all_ok,
            'time_ms': total_time,
            'explanation': f"Multi-domain pipeline: {len(results)} stages completed"
        }

    def _auto_decompose_problem(self, problem: ProblemSpec, data: Dict) -> ProblemSpec:
        key_to_type = {
            'packages': ProblemType.SORTING,
            'items': ProblemType.OPTIMIZATION,
            'cities': ProblemType.PATHFINDING,
            'graph': ProblemType.PATHFINDING,
            'capacities': ProblemType.OPTIMIZATION,
            'skus': ProblemType.SORTING,
            'warehouses': ProblemType.OPTIMIZATION,
            'countries': ProblemType.CLASSIFICATION,
            'disruptions': ProblemType.CLASSIFICATION,
            'suppliers': ProblemType.OPTIMIZATION,
            'inventory': ProblemType.OPTIMIZATION,
            'bids': ProblemType.OPTIMIZATION,
            'budget': ProblemType.OPTIMIZATION,
            'auctions': ProblemType.OPTIMIZATION,
            'sequence': ProblemType.SEARCH,
            'references': ProblemType.SEARCH,
            'alignments': ProblemType.SEARCH,
            'matrix': ProblemType.OPTIMIZATION,
            'vector': ProblemType.OPTIMIZATION,
            'tensor': ProblemType.OPTIMIZATION,
        }

        sub_problems = {}
        pipeline_order = []

        for key in data:
            if key in key_to_type:
                sub_problems[key] = ProblemSpec(
                    name=f"{problem.name}_{key}",
                    problem_type=key_to_type[key]
                )
                if key not in pipeline_order:
                    pipeline_order.append(key)

        if 'packages' in pipeline_order and 'cities' in pipeline_order:
            pipeline_order = ['packages', 'cities', 'capacities']

        problem.sub_problems = sub_problems
        problem.pipeline_order = pipeline_order
        logger.info("Auto-decomposed into: %s", pipeline_order)
        return problem

    def _solve_single_domain(self, problem_spec: ProblemSpec, data: Any,
                             use_llm: bool = False,
                             expected: Any = None) -> Dict:
        start = time.perf_counter()
        result_obj = Result()

        ptype = problem_spec.problem_type
        if ptype in (ProblemType.UNKNOWN, ProblemType.TRANSFORMATION, ProblemType.SORTING):
            detected = problem_spec.infer_problem_type(data)
            if detected != ptype:
                problem_spec.problem_type = detected

        data = self._prepare_input_data(problem_spec, data)

        context = self.context_engine.analyze(data, problem_spec.problem_type.value)
        context["data"] = data

        algo = self.meta_controller.select(
            context=context,
            candidates=list(self.registry.keys()),
            problem_spec=problem_spec,
        )

        if algo is None:
            algo = next(iter(self.registry.values()))
        algo_name = algo.name

        try:
            output = algo.process(data)
            success = algo.validate_output(data, output)
            if expected is not None and success:
                output_list = output if isinstance(output, list) else []
                expected_list = expected if isinstance(expected, list) else []
                success = (output_list == expected_list)
        except Exception as e:
            alt = self.meta_controller.get_fallback_alternative(algo_name)
            if alt and alt in self.registry:
                try:
                    algo = self.registry[alt]
                    algo_name = algo.name
                    output = algo.process(data)
                    success = True
                except Exception:
                    output = data
                    success = False
            else:
                output = data
                success = False

        elapsed = (time.perf_counter() - start) * 1000

        result_obj.output = output
        result_obj.algorithm = algo_name
        result_obj.time_ms = elapsed
        result_obj.success = success
        result_obj.pipeline = [algo_name]
        result_obj.metrics = {"wall_time_ms": elapsed, "success": success}

        self._execution_count += 1
        return result_obj.to_dict()

    def _prepare_input_data(self, problem_spec: ProblemSpec, raw_data: Any) -> Any:
        if raw_data is None:
            ptype = problem_spec.problem_type
            if ptype == ProblemType.SORTING:
                return []
            elif ptype == ProblemType.PATHFINDING:
                return {"graph": {}, "start": "", "end": ""}
            elif ptype == ProblemType.OPTIMIZATION:
                return {"items": [], "capacity": 0}
            return {}

        if isinstance(raw_data, str):
            try:
                import json
                return json.loads(raw_data)
            except Exception:
                try:
                    return eval(raw_data)
                except Exception:
                    return raw_data

        ptype = problem_spec.problem_type
        if ptype == ProblemType.PATHFINDING:
            if isinstance(raw_data, tuple) and len(raw_data) == 3:
                return {"graph": raw_data[0], "start": raw_data[1], "end": raw_data[2]}
            if isinstance(raw_data, dict):
                return raw_data
        elif ptype == ProblemType.OPTIMIZATION:
            if isinstance(raw_data, tuple) and len(raw_data) == 2:
                return {"items": raw_data[0], "capacity": raw_data[1]}
            if isinstance(raw_data, dict):
                return raw_data
        return raw_data

    def _build_registry(self) -> Dict:
        from algorithms.sorting import (
            QuickSort, TimSort, HeapSort, InsertionSort,
            RadixSort, MergeSort
        )
        from algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder
        from algorithms.optimization import GreedyKnapsack, SimulatedAnnealing
        from algorithms.ml import (
            KMeansClustering, DBSCANClustering,
            LinearRegression, RandomForestClassifier
        )
        from algorithms.ml.embeddings import (
            Word2VecTrainer, PCAReduction,
            TSNEVisualization, SemanticSimilarityGenerator
        )
        from algorithms.image_processing import (
            GaussianBlur, MedianFilter,
            BilateralFilter, SobelEdgeDetection,
            CLAHE
        )
        from algorithms.safety import (
            IdentityAlgorithm, SafeSort, SafePath,
            SafeKnapsack
        )

        registry = {}

        for cls in [IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack]:
            algo = cls(); registry[algo.name] = algo

        for cls in [QuickSort, TimSort, HeapSort, InsertionSort, RadixSort, MergeSort]:
            algo = cls(); registry[algo.name] = algo

        for cls in [Dijkstra, AStar, BFSPathfinder]:
            algo = cls(); registry[algo.name] = algo

        for cls in [GreedyKnapsack, SimulatedAnnealing]:
            algo = cls(); registry[algo.name] = algo

        for cls in [KMeansClustering, DBSCANClustering, LinearRegression, RandomForestClassifier]:
            algo = cls(); registry[algo.name] = algo

        for cls in [Word2VecTrainer, PCAReduction, TSNEVisualization, SemanticSimilarityGenerator]:
            algo = cls(); registry[algo.name] = algo

        for cls in [GaussianBlur, MedianFilter, BilateralFilter, SobelEdgeDetection, CLAHE]:
            algo = cls(); registry[algo.name] = algo

        return registry

    def _get_global_registry(self) -> Dict:
        return dict(self.registry)

    def explain_last(self) -> Optional[Dict]:
        sel = self.meta_controller.get_last_selection()
        if not sel:
            return None
        return {
            "algorithm": sel.get("algorithm"),
            "confidence": sel.get("confidence", 0),
        }

    def register_algorithm(self, algorithm):
        """Hot-register a new algorithm at runtime."""
        from algorithms.base import Algorithm
        if not isinstance(algorithm, Algorithm):
            raise TypeError("Must be instance of Algorithm base class")
        self.registry[algorithm.name] = algorithm
        logger.info("Registered algorithm: %s", algorithm.name)

    def register_from_file(self, file_path: str, class_name: str = None):
        """Load and register an algorithm from a .py file."""
        import importlib.util
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Algorithm file not found: {file_path}")

        spec = importlib.util.spec_from_file_location("custom_algo", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from algorithms.base import Algorithm
        found = False
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Algorithm) and obj != Algorithm:
                if class_name and name != class_name:
                    continue
                instance = obj()
                self.register_algorithm(instance)
                found = True

        if not found:
            raise ValueError(f"No Algorithm subclass found in {file_path}")

    def benchmark(self, data, spec=None):
        """Compare AAlgoI against standard library implementation."""
        import time as time_module

        if spec is None:
            spec = ProblemSpec(name="benchmark", problem_type=ProblemType.UNKNOWN)
            spec.problem_type = spec.infer_problem_type(data)

        start_aalgoi = time_module.time()
        result = self.solve(spec, data)
        aalgoi_time = time_module.time() - start_aalgoi

        start_baseline = time_module.time()
        baseline_time = self._run_baseline(data, spec)
        baseline_time = time_module.time() - start_baseline

        speedup = baseline_time / aalgoi_time if aalgoi_time > 0 else float("inf")

        return {
            "aalgoi_time_ms": aalgoi_time * 1000,
            "baseline_time_ms": baseline_time * 1000,
            "speedup_factor": round(speedup, 2),
            "aalgoi_algorithm": result.get("algorithm", "unknown"),
            "winner": "AAlgoI" if speedup > 1.05 else "Baseline",
        }

    def _run_baseline(self, data, spec):
        """Execute standard library implementation as baseline."""
        import time

        start = time.time()

        if spec.problem_type == ProblemType.SORTING:
            if isinstance(data, list):
                _ = sorted(data)
        elif spec.problem_type == ProblemType.OPTIMIZATION:
            pass
        elif spec.problem_type == ProblemType.PATHFINDING:
            pass

        return time.time() - start

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_solves": self._execution_count,
            "validator": self.validator.get_stats()
        }


class AAlgoI:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_components()
        self._execution_count: int = 0
        self._total_time_ms: float = 0.0
        self._reconfigure_threshold = self.config.get("reconfigure_threshold", 0.15)
        self._retrain_interval = self.config.get("retrain_interval", 100)

        self._baseline_times: Dict[str, List[float]] = {}
        self._hot_cache: Dict[str, Dict] = {}
        self._warm_cache_max = self.config.get("warm_cache", {}).get("max_entries", 1000)
        self._min_hits_for_hot = self.config.get("warm_cache", {}).get("min_hits_for_hot", 50)

        self._evolver = GeneticPipelineEvolver(
            self._build_algorithm_registry(),
            pop_size=self.config.get("genetic_pop_size", 20)
        )
        self._evolution_counter = 0

    def _init_components(self):
        self.context_engine = ContextEngine(config=self.config)

        registry = self._build_algorithm_registry()
        self.meta_controller = MetaController(
            algorithm_registry=registry,
            strategy=self.config.get("strategy", "hybrid"),
            config=self.config
        )

        self.compositor = DynamicCompositor(config=self.config)
        self.tracker = PerformanceTracker(config=self.config)

        self.knowledge = KnowledgeBase(config=self.config.get("knowledge_base", {}))
        self.meta_controller.knowledge = self.knowledge

        self.validator = PipelineValidator()
        self.drift_detector = DriftDetector(
            window=self.config.get("drift", {}).get("window", 100),
            threshold=self.config.get("drift", {}).get("threshold", 0.05)
        )
        self.decision_log = DecisionLog()

        self._active_pipeline: List[Any] = []
        self._active_graph: Optional[PipelineGraph] = None
        self._last_context: Optional[Dict] = None
        self._use_dag = self.config.get("enable_dag", True)

    def _build_algorithm_registry(self) -> Dict[str, Any]:
        from algorithms.sorting import (
            QuickSort, InsertionSort, MergeSort, TimSort, RadixSort, HeapSort
        )
        from algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder
        from algorithms.optimization import GreedyKnapsack, SimulatedAnnealing
        from algorithms.ml import (
            KMeansClustering, DBSCANClustering,
            LinearRegression, RandomForestClassifier
        )
        from algorithms.ml.embeddings import (
            Word2VecTrainer, PCAReduction,
            TSNEVisualization, SemanticSimilarityGenerator
        )
        from algorithms.image_processing import (
            GaussianBlur, MedianFilter,
            BilateralFilter, SobelEdgeDetection, CLAHE
        )
        from algorithms.safety import (
            IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack
        )

        registry = {}
        for cls in [IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack]:
            algo = cls(); registry[algo.name] = algo
        for cls in [QuickSort, InsertionSort, MergeSort, TimSort, RadixSort, HeapSort]:
            algo = cls(); registry[algo.name] = algo
        for cls in [GaussianBlur, MedianFilter, BilateralFilter, SobelEdgeDetection, CLAHE]:
            algo = cls(); registry[algo.name] = algo
        for cls in [KMeansClustering, DBSCANClustering, LinearRegression, RandomForestClassifier]:
            algo = cls(); registry[algo.name] = algo
        for cls in [Word2VecTrainer, PCAReduction, TSNEVisualization, SemanticSimilarityGenerator]:
            algo = cls(); registry[algo.name] = algo
        for cls in [Dijkstra, AStar, BFSPathfinder]:
            algo = cls(); registry[algo.name] = algo
        for cls in [GreedyKnapsack, SimulatedAnnealing]:
            algo = cls(); registry[algo.name] = algo
        return registry

    def run(self, data: Any, task_type: str = "auto", expected_result: Any = None) -> Any:
        start_total = time.perf_counter()

        context = self.context_engine.analyze(data, task_type=task_type)

        ctx_key = self._hash_context(context)
        hot_entry = self._hot_cache.get(ctx_key)

        if hot_entry and hot_entry.get("count", 0) >= self._min_hits_for_hot:
            self._active_pipeline = self._rebuild_pipeline_from_names(hot_entry["algo_names"])
            self._last_context = context
            context["algorithms"] = hot_entry["algo_names"]
        else:
            if self._should_reconfigure(context):
                algorithms = self.meta_controller.select(context)
                self._active_pipeline = self.compositor.build_pipeline(algorithms, context)
                self._active_graph = None
                self._last_context = context
                context["algorithms"] = [a.name for a in self._active_pipeline]

                if self._use_dag and len(self._active_pipeline) > 1:
                    graph = PipelineGraph()
                    for i, algo in enumerate(self._active_pipeline):
                        deps = [f"step_{i-1}"] if i > 0 else []
                        graph.add_algorithm(f"step_{i}", algo, depends_on=deps)
                    self._active_graph = graph

            self._update_hot_cache(ctx_key, context)

        if self._active_graph and self._use_dag:
            def pipeline_fn(d):
                return self._active_graph.execute(d)
        else:
            def pipeline_fn(d):
                result = d
                for algo in self._active_pipeline:
                    result = algo.process(result)
                return result

        result, metrics = self.tracker.evaluate(
            pipeline_fn, data, context, expected_result
        )

        validated_result = self._validate_and_fallback(
            result, data, context, metrics
        )

        algo_names = [a.name for a in self._active_pipeline]
        score = self._compute_composite_score(metrics, context, algo_names)

        metrics["score"] = score
        self.knowledge.store(context, algo_names, metrics)
        self.meta_controller.record(context, self._active_pipeline, score, metrics)

        success = metrics.get("success", True) and (validated_result is not None)
        if success and metrics.get("wall_time_ms", 0) > 0:
            throughput = len(data) / (metrics["wall_time_ms"] / 1000) if hasattr(data, '__len__') else 0
            bandit_reward = score * (0.5 + 0.5 * min(1.0, throughput / 1000))
            for algo_name in algo_names:
                self.meta_controller.bandit.update(algo_name, bandit_reward)

        self.meta_controller.bandit.decay_epsilon()

        drift_detected = self.drift_detector.update(score, time.time())
        if drift_detected:
            self.knowledge.discount_old_records(factor=0.5)
            self.meta_controller.bandit.reset_exploration()

        self._execution_count += 1
        if self._execution_count % self._retrain_interval == 0:
            self.meta_controller.train()

        self._evolution_counter += 1
        if self._evolution_counter >= 500:
            self._run_genetic_evolution()
            self._evolution_counter = 0

        self._total_time_ms += (time.perf_counter() - start_total) * 1000

        self.decision_log.record(Decision(
            context=context,
            candidates=list(self.meta_controller.registry.keys()),
            chosen=" → ".join(algo_names),
            confidence=self.meta_controller.get_last_confidence(),
            reason=self.meta_controller.get_last_reason(),
            outcome_success=success,
            wall_time_ms=metrics.get("wall_time_ms", 0)
        ))

        return validated_result

    def _rebuild_pipeline_from_names(self, algo_names: List[str]) -> List[Any]:
        pipeline = []
        registry = self.meta_controller.registry
        for name in algo_names:
            if name in registry:
                algo = registry[name].clone() if hasattr(registry[name], 'clone') else registry[name]
                pipeline.append(algo)
        return pipeline

    def _validate_and_fallback(self, result: Any, data: Any, context: Dict,
                                metrics: Dict) -> Any:
        if not metrics.get("success", True):
            safe_result, ok = self.meta_controller.execute_with_fallback(data, context, self._active_pipeline)
            if ok:
                return safe_result
            return data

        for algo in self._active_pipeline:
            if hasattr(algo, 'validate_output'):
                validation = algo.validate_output(data if self._active_pipeline.index(algo) == 0 else result, result)
                if not validation:
                    self.knowledge.penalize(algo.name, context)
                    safe_result, ok = self.meta_controller.execute_with_fallback(data, context, self._active_pipeline)
                    return safe_result if ok else data

        return result

    def _run_genetic_evolution(self):
        perf_data = self.meta_controller.get_stats().get("algorithm_performance", {})
        fitness = self._evolver.evaluate_fitness(perf_data)
        self._evolver.evolve(fitness)

    def _should_reconfigure(self, context: Dict) -> bool:
        if not self._last_context:
            return True

        current_size = context.get("data_profile", {}).get("size", 0)
        last_size = self._last_context.get("data_profile", {}).get("size", 0)

        if last_size > 0:
            size_delta_ratio = abs(current_size - last_size) / last_size
            if size_delta_ratio > self._reconfigure_threshold:
                return True
        elif current_size != last_size:
            return True

        current_budget = context.get("constraints", {}).get("time_budget_ms")
        last_budget = self._last_context.get("constraints", {}).get("time_budget_ms")
        if current_budget != last_budget:
            return True

        current_priority = context.get("constraints", {}).get("priority")
        last_priority = self._last_context.get("constraints", {}).get("priority")
        if current_priority != last_priority:
            return True

        current_patterns = context.get("data_profile", {}).get("patterns", {})
        last_patterns = self._last_context.get("data_profile", {}).get("patterns", {})

        if current_patterns.get("is_nearly_sorted") != last_patterns.get("is_nearly_sorted"):
            return True
        if current_patterns.get("is_sorted") != last_patterns.get("is_sorted"):
            return True

        return False

    def _hash_context(self, context: Dict) -> str:
        dp = context.get("data_profile", {})
        fp = context.get("features", {})
        cons = context.get("constraints", {})

        bucketed = {
            "size_bin": (dp.get("size", 0) or 0) // 1000,
            "type": dp.get("type", "unknown"),
            "sorted": dp.get("patterns", {}).get("is_sorted", False),
            "cpu_bin": int((fp.get("cpu_free", 0.5) or 0.5) * 5),
            "budget_bin": (cons.get("time_budget_ms", 500) or 500) // 50,
            "priority": cons.get("priority", "balanced"),
            "task": context.get("task_type", "sorting")
        }
        return str(bucketed)

    def _update_hot_cache(self, ctx_key: str, context: Dict):
        if ctx_key in self._hot_cache:
            self._hot_cache[ctx_key]["count"] += 1
        else:
            algo_names = [a.name for a in self._active_pipeline]
            self._hot_cache[ctx_key] = {
                "algo_names": algo_names,
                "count": 1,
                "last_seen": time.time()
            }

        if len(self._hot_cache) > self._warm_cache_max:
            oldest = min(self._hot_cache.keys(),
                        key=lambda k: self._hot_cache[k].get("last_seen", 0))
            del self._hot_cache[oldest]

    def _compute_composite_score(self, metrics: Dict, context: Dict, algo_names: List[str]) -> float:
        quality = metrics.get("quality_score", 0)
        within_budget = 1.0 if metrics.get("within_budget", True) else 0.0
        success = 1.0 if metrics.get("success", True) else 0.0

        time_budget = context.get("constraints", {}).get("time_budget_ms", 500)
        wall_time = metrics.get("wall_time_ms", time_budget)
        data_size = context.get("data_profile", {}).get("size", 1)

        throughput = data_size / wall_time if wall_time > 0 else 0
        baseline_throughput = 1000.0
        throughput_score = min(1.0, throughput / baseline_throughput)

        budget_score = 1.0 if wall_time <= time_budget else max(0, 1.0 - (wall_time - time_budget) / time_budget)

        priority = context.get("constraints", {}).get("priority", "balanced")

        if priority == "speed":
            weights = {"quality": 0.2, "throughput": 0.5, "budget": 0.2, "success": 0.1}
        elif priority == "accuracy":
            weights = {"quality": 0.6, "throughput": 0.1, "budget": 0.1, "success": 0.2}
        else:
            weights = {"quality": 0.3, "throughput": 0.3, "budget": 0.25, "success": 0.15}

        score = (
            weights["quality"] * quality +
            weights["throughput"] * throughput_score +
            weights["budget"] * budget_score +
            weights["success"] * success
        )

        return score

    def get_stats(self) -> Dict[str, Any]:
        return {
            "executions": self._execution_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / self._execution_count if self._execution_count > 0 else 0,
            "active_pipeline": [a.name for a in self._active_pipeline],
            "meta_controller": self.meta_controller.get_stats(),
            "knowledge_base": {
                "total_records": len(self.knowledge.records),
                "algorithm_stats": self.knowledge.get_all_stats()
            },
            "performance_tracker": {
                "total_measurements": len(self.tracker.measurements),
                "summaries": {
                    name: self.tracker.get_performance_summary(name)
                    for name in self.meta_controller.registry.keys()
                }
            },
            "drift": self.drift_detector.get_stats(),
            "decision_log": self.decision_log.get_stats(),
            "validator": self.validator.get_failure_stats(),
            "genetic_evolver": self._evolver.get_stats(),
            "hot_cache_size": len(self._hot_cache),
            "warm_cache_hits": sum(e["count"] for e in self._hot_cache.values())
        }

    def explain_decision(self) -> Dict[str, Any]:
        if not self._last_context:
            return {"error": "No execution yet"}

        context = self._last_context
        last_decision = self.decision_log.get_last()

        llm_explanation = ""
        if last_decision:
            algo_names = " → ".join([a.name for a in self._active_pipeline])
            result_metrics = self.tracker.measurements[-1]["metrics"] if self.tracker.measurements else {}
            try:
                llm_explanation = self.meta_controller.llm.explain_decision(
                    context, algo_names, result_metrics
                )
            except Exception:
                pass

        return {
            "context": {
                "data_size": context.get("data_profile", {}).get("size"),
                "data_type": context.get("data_profile", {}).get("type"),
                "patterns": context.get("data_profile", {}).get("patterns"),
                "cpu_free": context.get("features", {}).get("cpu_free"),
                "memory_free": context.get("features", {}).get("mem_free_ratio"),
                "time_budget": context.get("constraints", {}).get("time_budget_ms"),
                "priority": context.get("constraints", {}).get("priority")
            },
            "selected_algorithms": [a.name for a in self._active_pipeline],
            "algorithm_details": [a.describe() for a in self._active_pipeline],
            "reasoning": self._generate_reasoning(context),
            "confidence": self.meta_controller.get_last_confidence(),
            "selection_reason": self.meta_controller.get_last_reason(),
            "llm_explanation": llm_explanation if llm_explanation else "LLM not available"
        }

    def _generate_reasoning(self, context: Dict) -> List[str]:
        reasoning = []

        data_profile = context.get("data_profile", {})
        features = context.get("features", {})
        constraints = context.get("constraints", {})

        size = data_profile.get("size", 0)
        if size <= 10:
            reasoning.append(f"Data size ({size}) is very small -> insertion sort has lowest overhead")
        elif size > 100000:
            reasoning.append(f"Data size ({size}) is very large -> need most efficient algorithm")

        if data_profile.get("patterns", {}).get("is_nearly_sorted"):
            reasoning.append("Data is nearly sorted -> adaptive sort (timsort) is optimal")

        if data_profile.get("patterns", {}).get("is_sorted"):
            reasoning.append("Data is already sorted -> O(n) adaptive sort sufficient")

        cpu_free = features.get("cpu_free", 0)
        if cpu_free > 0.7:
            reasoning.append(f"CPU availability is high ({cpu_free:.1%}) -> can use compute-intensive algorithms")
        elif cpu_free < 0.3:
            reasoning.append(f"CPU is constrained ({cpu_free:.1%}) -> using lightweight approach")

        priority = constraints.get("priority", "balanced")
        if priority == "speed":
            reasoning.append("Priority is speed -> favoring fastest algorithms")
        elif priority == "accuracy":
            reasoning.append("Priority is accuracy -> favoring most reliable algorithms")

        confidence = self.meta_controller.get_last_confidence()
        reasoning.append(f"Decision confidence: {confidence:.0%} ({self.meta_controller.get_last_reason()})")

        return reasoning

    def benchmark(self, data: Any, algorithms: Optional[List[str]] = None) -> Dict[str, Dict]:
        if algorithms is None:
            algorithms = list(self.meta_controller.registry.keys())

        results = {}

        for algo_name in algorithms:
            if algo_name not in self.meta_controller.registry:
                continue

            algo = self.meta_controller.registry[algo_name]

            start = time.perf_counter()
            try:
                result = algo.process(data)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)

            elapsed_ms = (time.perf_counter() - start) * 1000

            is_sorted = False
            if isinstance(result, list) and len(result) > 1:
                is_sorted = all(result[i] <= result[i+1] for i in range(len(result)-1))

            results[algo_name] = {
                "time_ms": elapsed_ms,
                "success": success,
                "error": error,
                "correct": is_sorted,
                "result_sample": result[:10] if isinstance(result, list) and len(result) > 10 else result
            }

        return results

    def get_decision_log(self, n: int = 10) -> List[Dict]:
        decisions = self.decision_log.get_recent(n)
        return [{
            "chosen": d.chosen,
            "confidence": d.confidence,
            "reason": d.reason,
            "success": d.outcome_success,
            "time_ms": d.wall_time_ms,
            "timestamp": d.timestamp
        } for d in decisions]

    def get_drift_stats(self) -> Dict:
        return self.drift_detector.get_stats()

    def get_validation_stats(self) -> Dict:
        return self.validator.get_failure_stats()
