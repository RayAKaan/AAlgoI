
import json
import logging
import time
from typing import Any

import numpy as np

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
from aalgoi.core.compositor import DynamicCompositor
from aalgoi.core.context_engine import ContextEngine
from aalgoi.core.decision_log import Decision, DecisionLog
from aalgoi.core.drift_detector import DriftDetector
from aalgoi.core.explainer import Explainer
from aalgoi.core.genetic_evolver import GeneticPipelineEvolver
from aalgoi.core.knowledge_base import KnowledgeBase
from aalgoi.core.meta_controller import MetaController, UniversalMetaController
from aalgoi.core.performance_tracker import PerformanceTracker
from aalgoi.core.pipeline_graph import PipelineGraph
from aalgoi.core.problem_spec import ProblemSpec, ProblemType
from aalgoi.core.rl.reward_shaper import RewardShaper
from aalgoi.core.validator import LearningValidator, PipelineValidator

logger = logging.getLogger(__name__)


class Result:
    """
    Return type for all AAlgoI solve operations.
    Supports both dict-style and attribute-style access.
    Prints beautifully. Never silently returns None.
    """

    KNOWN_KEYS = {
        "result":      "result",
        "algorithm":   "algorithm",
        "time_ms":     "time_ms",
        "success":     "success",
        "answer":      "answer",
        "error":       "error",
        "confidence":  "confidence",
        "alternatives":"alternatives",
        "metrics":     "metrics",
        "pipeline":    "pipeline",
        # Aliases — map to canonical names
        "ok":          "success",
        "value":       "result",
        "ms":          "time_ms",
        "algo":        "algorithm",
        "output":      "result",
    }

    def __init__(
        self,
        result=None,
        algorithm: str = "",
        time_ms: float = 0.0,
        success: bool = True,
        answer: str = "",
        error: str = "",
        confidence: float = 0.0,
        alternatives: list = None,
        metrics: dict = None,
        pipeline: list = None,
    ):
        self.result = result
        self.algorithm = algorithm
        self.time_ms = time_ms
        self.success = success
        self.answer = answer
        self.error = error
        self.confidence = confidence
        self.alternatives = alternatives or []
        self.metrics = metrics or {}
        self.pipeline = pipeline or []

    # ── Dict-style access ──────────────────────────────────
    def __getitem__(self, key: str):
        canonical = self.KNOWN_KEYS.get(key)
        if canonical is None:
            raise KeyError(
                f"Result has no field '{key}'. "
                f"Valid keys: {sorted(k for k, v in Result.KNOWN_KEYS.items() if k == v)}"
            )
        return getattr(self, canonical)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return [k for k, v in self.KNOWN_KEYS.items() if k == v and hasattr(self, k)]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in [
            "result", "algorithm", "time_ms", "success",
            "answer", "error", "confidence", "alternatives",
            "metrics", "pipeline",
        ] if k in ("result", "algorithm", "time_ms", "success") or getattr(self, k) is not None}

    # ── Shortcuts ──────────────────────────────────────────
    @property
    def ok(self) -> bool:
        return self.success

    @property
    def value(self):
        return self.result

    @property
    def ms(self) -> float:
        return self.time_ms

    @property
    def algo(self) -> str:
        return self.algorithm

    # ── Display ───────────────────────────────────────────
    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        conf = f" ({self.confidence:.0%})" if self.confidence > 0 else ""
        return f"Result({status} {self.algorithm}{conf} in {self.time_ms:.2f}ms)"

    def __str__(self) -> str:
        if self.answer:
            return self.answer
        return repr(self)

    def __bool__(self) -> bool:
        return self.success

    def _repr_html_(self) -> str:
        status_color = "#2ecc71" if self.success else "#e74c3c"
        status_icon  = "✓" if self.success else "✗"
        return f"""
        <div style="font-family: monospace; padding: 8px; border-left: 3px solid {status_color};">
            <span style="color:{status_color}; font-weight:bold">{status_icon}</span>
            <strong>{self.algorithm}</strong>
            &nbsp;|&nbsp; {self.time_ms:.2f}ms
            &nbsp;|&nbsp; {f"{self.confidence:.0%} confidence" if self.confidence else ""}
            <br>
            <span style="color:#555">{self.answer or str(self.result)[:80]}</span>
        </div>
        """

    def __lt__(self, other) -> bool:
        return self.time_ms < other.time_ms


class UniversalSolver:
    def __init__(self, problem_library=None, llm_client=None, config=None):
        self.config = config or {}
        self.config.setdefault("rl_deterministic", False)
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

        self.reward_shaper = RewardShaper(config=self.config.get("reward", {}))
        self._solve_count = 0
        self._train_interval = self.config.get("train_interval", 64)
        self._total_solves = 0

        self._execution_count = 0

        # Push algorithm embeddings to RL agent's attention head
        from aalgoi.core.algorithm_embedder import AlgorithmEmbedder
        self.embedder = AlgorithmEmbedder()
        self.embedder.embed_all(self.registry)
        rl = self.meta_controller.rl_agent
        all_embeds = self.embedder.get_all_embeddings(self.registry)
        rl.update_algo_embeddings(all_embeds, list(self.registry.keys()))

        # LoRA adapter for per-user personalization
        from aalgoi.core.rl.lora_adapter import LoRAAdapter
        self.lora_adapter = LoRAAdapter(
            rl.network,
            rank=self.config.get('lora_rank', 4),
        )
        self.lora_adapter.apply()
        rl.lora_adapter = self.lora_adapter

        # Rebuild optimizer — LoRALinear base weights are frozen internally
        # so only trainable params (encoder, LoRA A/B, value_proj, critic) get gradients
        import torch as _torch
        rl.optimizer = _torch.optim.Adam(
            rl.network.parameters(),
            lr=rl.config.get("learning_rate", 3e-4),
        )

        # Checkpoint manager for versioned LoRA weights
        from aalgoi.core.checkpoint_manager import CheckpointManager
        self.checkpoint_manager = CheckpointManager()
        adapter_path = self.checkpoint_manager.get_current_adapter_path()
        if adapter_path:
            self.lora_adapter.load(adapter_path)

        # GitHub registry sync (background thread)
        from aalgoi.core.registry_sync import GitHubRegistrySync
        self.registry_sync = GitHubRegistrySync(
            local_registry=self.registry,
            embedder=self.embedder,
            agent=rl,
        )
        self.registry_sync.start()

        from aalgoi.core.checkpoint_downloader import ensure_checkpoint_async
        ensure_checkpoint_async()

        self._checkpoint_interval = self.config.get('checkpoint_interval', 50)

    def solve(self, problem_spec: ProblemSpec, data: Any,
              use_llm: bool = False,
              expected: Any = None) -> dict:
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

    def _solve_multi_domain(self, problem: ProblemSpec, data: dict) -> dict[str, Any]:
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

    def _auto_decompose_problem(self, problem: ProblemSpec, data: dict) -> ProblemSpec:
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
                             expected: Any = None) -> dict:
        start = time.perf_counter()

        ptype = problem_spec.problem_type

        # Only override with data shape inference when NL parser was uncertain
        if data is not None and ptype in (ProblemType.UNKNOWN, ProblemType.TRANSFORMATION):
            from_data = ProblemSpec._infer_from_data_shape(data)
            if from_data != ProblemType.UNKNOWN:
                ptype = from_data
                problem_spec.problem_type = from_data

        if ptype in (ProblemType.UNKNOWN, ProblemType.TRANSFORMATION, ProblemType.SORTING):
            detected = problem_spec.infer_problem_type(data)
            if detected != ptype:
                problem_spec.problem_type = detected

        data = self._prepare_input_data(data, problem_spec.problem_type.value)

        context = self.context_engine.analyze(data, problem_spec.problem_type.value)
        context["data"] = data

        query = problem_spec.description or problem_spec.name or ""
        algo = self.meta_controller.select(
            context=context,
            candidates=list(self.registry.keys()),
            problem_spec=problem_spec,
            query=query,
        )

        if algo is None:
            algo = next(iter(self.registry.values()))
        algo_name = algo.name

        try:
            output = algo.process(data)
            success = algo.validate_output(data, output)
            if success:
                from aalgoi.core.oracles import evaluate as oracle_evaluate
                if not oracle_evaluate(problem_spec.problem_type, data, output):
                    logger.warning("Oracle rejected %s output for %s", algo_name, problem_spec.problem_type)
                    success = False
            output = self._normalize_result(output, algo_name)
            if expected is not None and success:
                output_list = output if isinstance(output, list) else []
                expected_list = expected if isinstance(expected, list) else []
                success = (output_list == expected_list)
        except Exception:
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

        elapsed_ms = (time.perf_counter() - start) * 1000

        # === ONLINE RL FEEDBACK LOOP ===
        state = self.meta_controller._last_state
        action_idx = self.meta_controller._last_action_idx
        log_prob = self.meta_controller._last_log_prob
        value = self.meta_controller._last_value

        if state is not None and action_idx is not None:
            data_size = len(data) if hasattr(data, '__len__') else 0
            reward = self.reward_shaper.compute(
                success=success,
                elapsed=elapsed_ms / 1000.0,
                data_size=data_size,
                algo_name=algo_name,
            )
            agent = self.meta_controller.rl_agent
            agent.store_transition(
                state=state,
                action=action_idx,
                reward=reward,
                done=True,
                log_prob=log_prob,
                value=value,
            )

            self._solve_count += 1
            self._total_solves += 1
            if self._solve_count >= self._train_interval:
                train_result = agent.train()
                if train_result and train_result.get("total_loss", 0.0) > 0.0:
                    logger.info(
                        f"RL trained: loss={train_result['total_loss']:.4f}, "
                        f"policy_loss={train_result['policy_loss']:.4f}, "
                        f"entropy={train_result['entropy']:.4f}, "
                        f"buffer={len(agent.buffer)}"
                    )
                self._solve_count = 0

            # Save LoRA checkpoint periodically
            if hasattr(self, 'checkpoint_manager') and self._total_solves % self._checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    adapter=self.lora_adapter,
                    solve_count=self._total_solves,
                    metrics={'success_rate': float(success), 'reward': reward},
                )
        # === END FEEDBACK LOOP ===

        result_obj = Result(
            result=output,
            algorithm=algo_name,
            time_ms=elapsed_ms,
            success=success,
            pipeline=[algo_name],
            metrics={"wall_time_ms": elapsed_ms, "success": success},
        )

        self._execution_count += 1
        return result_obj.to_dict()

    def _prepare_input_data(self, data, algo_name: str = "") -> Any:
        # ML data — pass through untouched with numpy conversion
        if isinstance(data, dict):
            ml_signal_keys = {
                "X_train", "y_train", "X_test",
                "n_clusters", "n_components",
                "corpus", "texts", "text", "query", "document",
                "words", "seed_word", "operation", "num_sentences",
            }
            if any(k in data for k in ml_signal_keys):
                prepared = {}
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        prepared[k] = v.tolist()
                    elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                        prepared[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
                    else:
                        prepared[k] = v
                return prepared

            # Graph/image/optimization data — pass through as-is
            return data

        # Non-dict data — handle None and string parsing
        if data is None:
            ptype = algo_name
            if ptype == "sorting":
                return []
            elif ptype == "pathfinding":
                return {"graph": {}, "start": "", "end": ""}
            elif ptype == "optimization":
                return {"items": [], "capacity": 0}
            return {}

        if isinstance(data, str):
            try:
                return json.loads(data)
            except Exception:
                try:
                    return eval(data)
                except Exception:
                    return data

        return data

    def _normalize_result(self, result: Any, algo_name: str) -> dict:
        if isinstance(result, dict):
            ml_classifiers = {
                "knn_classifier", "decision_tree_classifier", "svm_classifier",
                "naive_bayes_classifier", "logistic_regression",
                "random_forest_classifier", "xgboost_classifier",
            }
            ml_regressors = {
                "linear_regression", "polynomial_regression", "ridge_regression",
                "lasso_regression", "decision_tree_regressor", "random_forest_regressor",
            }
            ml_clustering = {
                "kmeans", "dbscan", "hierarchical_clustering", "gmm",
            }
            ml_reduction = {
                "pca_reduction", "tsne_reduction",
            }

            if algo_name in ml_classifiers and "predictions" not in result:
                for alt_key in ("preds", "y_pred", "y_test", "predicted", "labels", "result", "output"):
                    if alt_key in result:
                        val = result[alt_key]
                        if isinstance(val, (list, np.ndarray)):
                            result["predictions"] = list(val) if isinstance(val, np.ndarray) else val
                            break

            if algo_name in ml_regressors and "predictions" not in result:
                for alt_key in ("preds", "y_pred", "y_test", "predicted", "values", "result", "output"):
                    if alt_key in result:
                        val = result[alt_key]
                        if isinstance(val, (list, np.ndarray)):
                            result["predictions"] = list(val) if isinstance(val, np.ndarray) else val
                            break

            if algo_name in ml_clustering and "labels" not in result:
                for alt_key in ("clusters", "cluster_labels", "assignments", "result", "output", "predictions"):
                    if alt_key in result:
                        val = result[alt_key]
                        if isinstance(val, (list, np.ndarray)):
                            result["labels"] = list(val) if isinstance(val, np.ndarray) else val
                            break

            if algo_name in ml_reduction and "transformed" not in result:
                for alt_key in ("components", "embedding", "reduced", "result", "output", "X_transformed"):
                    if alt_key in result:
                        val = result[alt_key]
                        if isinstance(val, (list, np.ndarray)):
                            result["transformed"] = list(val) if isinstance(val, np.ndarray) else val.tolist() if isinstance(val, np.ndarray) else val
                            break

            return result

        pathfinding_names = {
            "bfs", "dfs", "dijkstra", "astar", "bellman_ford",
            "floyd_warshall", "greedy_search",
        }

        if isinstance(result, (list, tuple)):
            if algo_name in pathfinding_names:
                return {"path": list(result), "cost": None}
            return result

        if isinstance(result, np.ndarray):
            return {"result": result.tolist()}

        if isinstance(result, (int, float, str, bool)):
            return {"result": result}

        return {"result": result}

    def _build_registry(self) -> dict:
        from aalgoi.algorithms.image_processing import (
            CLAHE,
            BilateralFilter,
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
            GaussianNBAlgo,
            GMMAlgo,
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
        from aalgoi.algorithms.sorting import HeapSort, InsertionSort, MergeSort, QuickSort, RadixSort, TimSort

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

        def _register_algo(algo):
            self._validate_name(algo.name)
            if algo.name in registry:
                raise KeyError(f"Duplicate registration: '{algo.name}'")
            registry[algo.name] = algo

        for cls in [IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack]:
            _register_algo(cls())

        for cls in [QuickSort, TimSort, HeapSort, InsertionSort, RadixSort, MergeSort]:
            _register_algo(cls())

        for cls in [Dijkstra, AStar, BFSPathfinder, FloydWarshall]:
            _register_algo(cls())

        for cls in [GreedyKnapsack, SimulatedAnnealing, GeneticAlgorithm,
                     HillClimbing, ParticleSwarmOptimization, AntColonyOptimization]:
            _register_algo(cls())

        for cls in ml_classes:
            try:
                _register_algo(cls())
            except ImportError as e:
                logger.info("Skipping %s: %s", cls.__name__, e)
            except Exception as e:
                logger.warning("Failed to load %s: %s", cls.__name__, e)

        for cls in [PCAReduction, TSNEVisualization, SemanticSimilarityGenerator]:
            _register_algo(cls())

        for cls in nlp_classes:
            try:
                _register_algo(cls())
            except Exception as e:
                logger.info("Skipping %s: %s", cls.__name__, e)

        for cls in [GaussianBlur, MedianFilter, BilateralFilter, SobelEdgeDetection, CLAHE,
                     CannyEdgeDetection, LaplacianEdgeDetection, NLMDenoising, MorphologyOperation]:
            _register_algo(cls())

        return registry

    def _get_global_registry(self) -> dict:
        return dict(self.registry)

    def explain_last(self) -> dict | None:
        sel = self.meta_controller.get_last_selection()
        if not sel:
            return None
        return {
            "algorithm": sel.get("algorithm"),
            "confidence": sel.get("confidence", 0),
        }

    @staticmethod
    def _validate_name(name: str) -> None:
        import re
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            raise ValueError(
                f"Invalid algorithm name '{name}': must be snake_case "
                f"(lowercase letters, digits, underscores only)"
            )

    def register_algorithm(self, algorithm):
        """Hot-register a new algorithm at runtime.
        Immediately generates its embedding and updates
        the RL agent's key matrix. No retraining required.
        """
        from aalgoi.algorithms.base import Algorithm
        if not isinstance(algorithm, Algorithm):
            raise TypeError("Must be instance of Algorithm base class")
        self._validate_name(algorithm.name)
        if algorithm.name in self.registry:
            raise KeyError(f"Duplicate algorithm registration: '{algorithm.name}'")
        self.registry[algorithm.name] = algorithm
        from aalgoi.core.algorithm_embedder import AlgorithmEmbedder
        if not hasattr(self, 'embedder'):
            self.embedder = AlgorithmEmbedder()
            self.embedder.embed_all(self.registry)
        else:
            self.embedder.embed_algorithm(algorithm)
        rl = self.meta_controller.rl_agent
        all_embeds = self.embedder.get_all_embeddings(self.registry)
        rl.update_algo_embeddings(all_embeds, list(self.registry.keys()))
        logger.info("Registered algorithm: %s (embeddings updated)", algorithm.name)

    def register_from_file(self, file_path: str, class_name: str = None):
        """Load and register an algorithm from a .py file."""
        import importlib.util
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Algorithm file not found: {file_path}")

        spec = importlib.util.spec_from_file_location("custom_algo", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from aalgoi.algorithms.base import Algorithm
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

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_solves": self._execution_count,
            "validator": self.validator.get_stats()
        }


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

        self._active_pipeline: list[Any] = []
        self._active_graph: PipelineGraph | None = None
        self._last_context: dict | None = None
        self._use_dag = self.config.get("enable_dag", True)

    def _build_algorithm_registry(self) -> dict[str, Any]:
        from aalgoi.algorithms.image_processing import (
            CLAHE,
            BilateralFilter,
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
            GaussianNBAlgo,
            GMMAlgo,
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
        from aalgoi.algorithms.sorting import HeapSort, InsertionSort, MergeSort, QuickSort, RadixSort, TimSort

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

    def _run_genetic_evolution(self):
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

    def _update_hot_cache(self, ctx_key: str, context: dict):
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
        1.0 if metrics.get("within_budget", True) else 0.0
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
