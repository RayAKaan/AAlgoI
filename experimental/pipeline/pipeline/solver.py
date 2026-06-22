"""UniversalSolver — core solve engine."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import time
from typing import Any

import numpy as np

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
from aalgoi.core.checkpoint_downloader import ensure_checkpoint_async
from aalgoi.core.checkpoint_manager import CheckpointManager
from aalgoi.core.context_engine import ContextEngine
from aalgoi.core.explainer import Explainer
from aalgoi.core.oracles import evaluate as oracle_evaluate
from aalgoi.core.problem_spec import ProblemSpec, ProblemType
from aalgoi.core.registry_sync import GitHubRegistrySync
from aalgoi.core.rl.reward_shaper import RewardShaper
from aalgoi.core.meta_controller import UniversalMetaController
from aalgoi.core.validator import LearningValidator
from aalgoi.pipeline.result import Result

logger = logging.getLogger(__name__)


class UniversalSolver:
    def __init__(self, problem_library: Any = None, llm_client: Any = None, config: Any = None) -> None:
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

        self.embedder = None
        self.lora_adapter = None

        try:
            from aalgoi.core.algorithm_embedder import AlgorithmEmbedder
            self.embedder = AlgorithmEmbedder()
            self.embedder.embed_all(self.registry)
            rl = self.meta_controller.rl_agent
            if rl is not None:
                all_embeds = self.embedder.get_all_embeddings(self.registry)
                rl.update_algo_embeddings(all_embeds, list(self.registry.keys()))

                from aalgoi.core.rl.lora_adapter import LoRAAdapter
                self.lora_adapter = LoRAAdapter(
                    rl.network,
                    rank=self.config.get('lora_rank', 4),
                )
                self.lora_adapter.apply()
                rl.lora_adapter = self.lora_adapter

                import torch as _torch
                rl.optimizer = _torch.optim.Adam(
                    rl.network.parameters(),
                    lr=rl.config.get("learning_rate", 3e-4),
                )
        except ImportError:
            logger.info("RL features disabled: torch not installed")

        self.checkpoint_manager = CheckpointManager()
        adapter_path = self.checkpoint_manager.get_current_adapter_path()
        if adapter_path and self.lora_adapter is not None:
            self.lora_adapter.load(adapter_path)

        self.registry_sync = GitHubRegistrySync(
            local_registry=self.registry,
            embedder=self.embedder,
            agent=getattr(self.meta_controller, "rl_agent", None),
        )
        self.registry_sync.start()

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

            if hasattr(self, 'checkpoint_manager') and self._total_solves % self._checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    adapter=self.lora_adapter,
                    solve_count=self._total_solves,
                    metrics={'success_rate': float(success), 'reward': reward},
                )

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

    def _prepare_input_data(self, data: Any, algo_name: str = "") -> Any:
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

            return data

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

        def _register_algo(algo: Any) -> None:
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
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            raise ValueError(
                f"Invalid algorithm name '{name}': must be snake_case "
                f"(lowercase letters, digits, underscores only)"
            )

    def register_algorithm(self, algorithm: Any) -> None:
        """Hot-register a new algorithm at runtime."""
        from aalgoi.algorithms.base import Algorithm
        if not isinstance(algorithm, Algorithm):
            raise TypeError("Must be instance of Algorithm base class")
        self._validate_name(algorithm.name)
        if algorithm.name in self.registry:
            raise KeyError(f"Duplicate algorithm registration: '{algorithm.name}'")
        self.registry[algorithm.name] = algorithm
        from aalgoi.core.algorithm_embedder import AlgorithmEmbedder
        if not hasattr(self, 'embedder') or self.embedder is None:
            self.embedder = AlgorithmEmbedder()
            self.embedder.embed_all(self.registry)
        else:
            self.embedder.embed_algorithm(algorithm)
        rl = self.meta_controller.rl_agent
        all_embeds = self.embedder.get_all_embeddings(self.registry)
        rl.update_algo_embeddings(all_embeds, list(self.registry.keys()))
        logger.info("Registered algorithm: %s (embeddings updated)", algorithm.name)

    def register_from_file(self, file_path: str, class_name: str | None = None) -> None:
        """Load and register an algorithm from a .py file."""
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

    def benchmark(self, data: Any, spec: Any = None) -> dict:
        """Compare AAlgoI against standard library implementation."""
        if spec is None:
            spec = ProblemSpec(name="benchmark", problem_type=ProblemType.UNKNOWN)
            spec.problem_type = spec.infer_problem_type(data)

        start_aalgoi = time.time()
        result = self.solve(spec, data)
        aalgoi_time = time.time() - start_aalgoi

        start_baseline = time.time()
        baseline_time = self._run_baseline(data, spec)
        baseline_time = time.time() - start_baseline

        speedup = baseline_time / aalgoi_time if aalgoi_time > 0 else float("inf")

        return {
            "aalgoi_time_ms": aalgoi_time * 1000,
            "baseline_time_ms": baseline_time * 1000,
            "speedup_factor": round(speedup, 2),
            "aalgoi_algorithm": result.get("algorithm", "unknown"),
            "winner": "AAlgoI" if speedup > 1.05 else "Baseline",
        }

    def _run_baseline(self, data: Any, spec: Any) -> float:
        """Execute standard library implementation as baseline."""
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
