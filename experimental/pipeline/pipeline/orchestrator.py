"""AAlgoI orchestrator — pipeline orchestration and decision system."""

from __future__ import annotations

import logging
import time
from typing import Any

from aalgoi.algorithms.image_processing import (
    BilateralFilter,
    CLAHE,
    CannyEdgeDetection,
    GaussianBlur,
    LaplacianEdgeDetection,
    MedianFilter,
    MorphologyOperation,
    NLMDenoising,
    SobelEdgeDetection,
)
from aalgoi.algorithms.ml import (
    DBSCANClustering,
    GMMAlgo,
    GaussianNBAlgo,
    KMeansClustering,
    KNNAlgo,
    LassoAlgo,
    LightGBMAlgo,
    LinearRegressionAlgo,
    LogisticRegressionAlgo,
    PCAReductionAlgo,
    RandomForestAlgo,
    RidgeAlgo,
    SVMAlgo,
    XGBoostAlgo,
)
from aalgoi.algorithms.ml.embeddings import PCAReduction, SemanticSimilarityGenerator, TSNEVisualization
from aalgoi.algorithms.nlp import (
    CreativeSentenceGenerator,
    EmbeddingVisualization,
    FrequencyVectorArithmetic,
    PromptEnricher,
    RAGRetriever,
    SemanticSearcher,
    SentimentAnalyzer,
    TextSummarizer,
    Word2VecTrainer,
    WordExpander,
    WordVectorArithmetic,
)
from aalgoi.algorithms.optimization import (
    AntColonyOptimization,
    GeneticAlgorithm,
    GreedyKnapsack,
    HillClimbing,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
)
from aalgoi.algorithms.pathfinding import AStar, BFSPathfinder, Dijkstra, FloydWarshall
from aalgoi.algorithms.safety import IdentityAlgorithm, SafeKnapsack, SafePath, SafeSort
from aalgoi.algorithms.sorting import HeapSort, InsertionSort, MergeSort, QuickSort, RadixSort, TimSort
from aalgoi.core.compositor import DynamicCompositor
from aalgoi.core.context_engine import ContextEngine
from aalgoi.core.decision_log import Decision, DecisionLog
from aalgoi.core.drift_detector import DriftDetector
from aalgoi.core.genetic_evolver import GeneticPipelineEvolver
from aalgoi.core.knowledge_base import KnowledgeBase
from aalgoi.core.meta_controller import MetaController
from aalgoi.core.performance_tracker import PerformanceTracker
from aalgoi.core.pipeline_graph import PipelineGraph
from aalgoi.core.validator import PipelineValidator

logger = logging.getLogger(__name__)


class AAlgoI:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._init_components()
        self._execution_count: int = 0
        self._total_time_ms: float = 0.0
        self._reconfigure_threshold = self.config.get("reconfigure_threshold", 0.15)
        self._retrain_interval = self.config.get("retrain_interval", 100)

        self._baseline_times: dict[str, list[float]] = {}
        self._hot_cache: dict[str, dict] = {}
        self._warm_cache_max = self.config.get("warm_cache", {}).get("max_entries", 1000)
        self._min_hits_for_hot = self.config.get("warm_cache", {}).get("min_hits_for_hot", 50)

        self._evolver = GeneticPipelineEvolver(
            self._build_algorithm_registry(),
            pop_size=self.config.get("genetic_pop_size", 20)
        )
        self._evolution_counter = 0

    def _init_components(self) -> None:
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

        self._active_pipeline: list[Any] = []
        self._active_graph: PipelineGraph | None = None
        self._last_context: dict | None = None
        self._use_dag = self.config.get("enable_dag", True)

    def _build_algorithm_registry(self) -> dict[str, Any]:
        ml_classes = [
            LinearRegressionAlgo,
            RidgeAlgo,
            LassoAlgo,
            LogisticRegressionAlgo,
            KNNAlgo,
            SVMAlgo,
            GaussianNBAlgo,
            RandomForestAlgo,
            XGBoostAlgo,
            LightGBMAlgo,
            KMeansClustering,
            DBSCANClustering,
            GMMAlgo,
            PCAReductionAlgo,
        ]

        nlp_classes = [
            Word2VecTrainer,
            FrequencyVectorArithmetic,
            WordVectorArithmetic,
            EmbeddingVisualization,
            SentimentAnalyzer,
            TextSummarizer,
            RAGRetriever,
            SemanticSearcher,
            PromptEnricher,
            CreativeSentenceGenerator,
            WordExpander,
        ]

        registry = {}
        for cls in [IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack]:
            algo = cls(); registry[algo.name] = algo
        for cls in [QuickSort, InsertionSort, MergeSort, TimSort, RadixSort, HeapSort]:
            algo = cls(); registry[algo.name] = algo
        for cls in [GaussianBlur, MedianFilter, BilateralFilter, SobelEdgeDetection, CLAHE,
                     CannyEdgeDetection, LaplacianEdgeDetection, NLMDenoising, MorphologyOperation]:
            algo = cls(); registry[algo.name] = algo
        for cls in ml_classes:
            try:
                algo = cls()
                registry[algo.name] = algo
            except ImportError as e:
                logger.info("Skipping %s: %s", cls.__name__, e)
            except Exception as e:
                logger.warning("Failed to load %s: %s", cls.__name__, e)
        for cls in [PCAReduction, TSNEVisualization, SemanticSimilarityGenerator]:
            algo = cls(); registry[algo.name] = algo
        for cls in nlp_classes:
            try:
                algo = cls()
                registry[algo.name] = algo
            except Exception as e:
                logger.info("Skipping %s: %s", cls.__name__, e)
        for cls in [Dijkstra, AStar, BFSPathfinder, FloydWarshall]:
            algo = cls(); registry[algo.name] = algo
        for cls in [GreedyKnapsack, SimulatedAnnealing, GeneticAlgorithm,
                     HillClimbing, ParticleSwarmOptimization, AntColonyOptimization]:
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
            def pipeline_fn(d: Any) -> Any:
                return self._active_graph.execute(d)
        else:
            def pipeline_fn(d: Any) -> Any:
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

    def _rebuild_pipeline_from_names(self, algo_names: list[str]) -> list[Any]:
        pipeline = []
        registry = self.meta_controller.registry
        for name in algo_names:
            if name in registry:
                algo = registry[name].clone() if hasattr(registry[name], 'clone') else registry[name]
                pipeline.append(algo)
        return pipeline

    def _validate_and_fallback(self, result: Any, data: Any, context: dict,
                                metrics: dict) -> Any:
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

    def _run_genetic_evolution(self) -> None:
        perf_data = self.meta_controller.get_stats().get("algorithm_performance", {})
        fitness = self._evolver.evaluate_fitness(perf_data)
        self._evolver.evolve(fitness)

    def _should_reconfigure(self, context: dict) -> bool:
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

    def _hash_context(self, context: dict) -> str:
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

    def _update_hot_cache(self, ctx_key: str, context: dict) -> None:
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

    def _compute_composite_score(self, metrics: dict, context: dict, algo_names: list[str]) -> float:
        quality = metrics.get("quality_score", 0)
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

    def get_stats(self) -> dict[str, Any]:
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

    def explain_decision(self) -> dict[str, Any]:
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

    def _generate_reasoning(self, context: dict) -> list[str]:
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

    def benchmark(self, data: Any, algorithms: list[str] | None = None) -> dict[str, dict]:
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

    def get_decision_log(self, n: int = 10) -> list[dict]:
        decisions = self.decision_log.get_recent(n)
        return [{
            "chosen": d.chosen,
            "confidence": d.confidence,
            "reason": d.reason,
            "success": d.outcome_success,
            "time_ms": d.wall_time_ms,
            "timestamp": d.timestamp
        } for d in decisions]

    def get_drift_stats(self) -> dict:
        return self.drift_detector.get_stats()

    def get_validation_stats(self) -> dict:
        return self.validator.get_failure_stats()
