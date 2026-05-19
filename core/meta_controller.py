
import numpy as np
import time
import random
import os
import glob as glob_mod
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

from core.bandit import UCB1Bandit
from core.llm_adapter import LLMAdapter
from core.problem_spec import ProblemSpec, ProblemType
from core.algorithm_synthesizer import AlgorithmSynthesizer
from core.knowledge_graph import AlgorithmKnowledgeGraph
from algorithms.primitives import PRIMITIVES

logger = logging.getLogger(__name__)


class UniversalMetaController:
    def __init__(self, config: Dict = None, knowledge_base=None,
                 algorithm_registry: Dict = None,
                 problem_library=None, llm_client=None):
        self.config = config or {}
        self.kb = knowledge_base
        self.registry = algorithm_registry or {}
        self.history = []
        self._last_selection = None
        self._last_confidence = 0.0

        # Configurable features (off by default for minimal MVP)
        self.use_bandit = self.config.get('use_bandit', False)
        self.use_drift = self.config.get('use_drift', False)
        self.use_genetic = self.config.get('use_genetic', False)

        # RL Agent (core) — shared with RLPretrainer
        self.rl_agent = None
        self._rl_episode_buffer = []
        self._init_rl()

        self.fallback_chain = self._build_fallback_chain()
        # Knowledge Graph
        self.kg = AlgorithmKnowledgeGraph()
        self.kg_enabled = self.config.get('kg_enabled', True)
        self._build_knowledge_graph()

        # Load pre-trained weights
        self._load_pretrained_model()

        logger.info(
            f"UniversalMetaController initialized: "
            f"{len(self.registry)} algorithms, "
            f"RL={'ready' if self.rl_agent else 'no'}, "
            f"KG={self.kg_enabled}"
        )

    def _init_rl(self):
        try:
            from core.rl.agents.selection_agent import PPOAgent
            num_actions = max(len(self.registry), 1)
            rl_config = self.config.get("rl", {}).get("agents", {}).get("selection", {})
            self.rl_agent = PPOAgent(
                state_dim=self.config.get("rl", {}).get("state_dim", 200),
                num_actions=num_actions,
                config=rl_config,
            )
            self._rl_episode_buffer = []
            logger.info(f"RL agent created: state_dim=200, actions={num_actions}")
        except Exception as e:
            logger.warning(f"Failed to init RL agent: {e}")

    def _load_pretrained_model(self):
        model_pattern = self.config.get(
            'pretrained_model_path',
            "checkpoints/pretrain_v2/best_model_ep*.pt"
        )
        model_files = glob_mod.glob(model_pattern)
        if not model_files:
            v12_path = "checkpoints/pretrained_v1.pt"
            if os.path.exists(v12_path):
                model_files = [v12_path]
        if model_files:
            latest = max(model_files, key=os.path.getctime)
            try:
                self.rl_agent.load(latest)
                logger.info(f"Loaded pre-trained RL weights: {latest}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained model {latest}: {e}")
        else:
            logger.info("No pretrained model found (fresh RL agent)")

    def select(self, context: dict, candidates: list = None,
               problem_spec: ProblemSpec = None):
        candidates = candidates or []
        if not candidates and self.registry:
            candidates = list(self.registry.keys())
        if problem_spec is None:
            task_type = context.get("task_type", "sorting")
            problem_spec = ProblemSpec(
                name="auto",
                problem_type=ProblemType(task_type)
                if task_type in [t.value for t in ProblemType]
                else ProblemType.UNKNOWN
            )

        # 1. KG narrows candidates by problem type + constraints
        kg_pool = None
        if self.kg_enabled and problem_spec:
            constraints = self._extract_kg_constraints(problem_spec, context.get("data", {}))
            semantic_candidates = self.kg.find_candidates(
                problem_spec.problem_type.value, constraints
            )
            if semantic_candidates:
                candidate_names = {c if isinstance(c, str) else c.get("name", "") for c in candidates}
                filtered = [c for c in candidates
                            if (c if isinstance(c, str) else c.get("name", "")) in semantic_candidates]
                if filtered:
                    kg_pool = filtered
                    candidates = filtered

        # Ensure candidates is non-empty before indexing
        if not candidates:
            fallbacks = self._get_fallback_algorithms()
            if fallbacks:
                candidates = [fb.name for fb in fallbacks]
            if not candidates and self.registry:
                candidates = list(self.registry.keys())
            if not candidates:
                return None

        # 2. RL picks, validated against KG pool
        if self.rl_agent is not None:
            state = self._build_state_vector(context, problem_spec)
            action_idx, _, value = self.rl_agent.select_action(state, deterministic=True)
            raw_name = list(self.registry.keys())[action_idx % len(self.registry)]

            if kg_pool and raw_name not in kg_pool:
                best_kg = kg_pool[0] if isinstance(kg_pool[0], str) else kg_pool[0].get("name", "")
                algo_name = best_kg if best_kg in self.registry else raw_name
            else:
                algo_name = raw_name

            self._last_confidence = min(1.0, max(0.0, float(value)))
            if algo_name in self.registry:
                self._last_selection = {"algorithm": algo_name, "confidence": self._last_confidence}
                return self.registry[algo_name]

        # 3. Single-algorithm guarantee: always return one Algorithm, never a list
        pick = candidates[0] if isinstance(candidates[0], str) else candidates[0].get("name", "")
        if pick in self.registry:
            return self.registry[pick]

        fallbacks = self._get_fallback_algorithms()
        if fallbacks:
            return fallbacks[0]
        if self.registry:
            return list(self.registry.values())[0]
        return None

    def get_fallback_alternative(self, failed_algo_name: str) -> Optional[str]:
        if not self.kg_enabled:
            return None
        alternatives = self.kg.find_alternatives(failed_algo_name)
        return alternatives[0] if alternatives else None

    def _build_fallback_chain(self) -> List[str]:
        """Ordered list of safe algorithms to try when all else fails."""
        return [
            "identity",
            "safe_sort",
            "quicksort",
            "bfs_path",
            "greedy_knapsack",
        ]

    def _get_fallback_algorithms(self) -> List[Any]:
        """Return fallback algorithms from the chain that exist in registry."""
        fallbacks = []
        for name in self.fallback_chain:
            if name in self.registry:
                fallbacks.append(self.registry[name])
        return fallbacks

    def _extract_kg_constraints(self, spec: ProblemSpec, data: Any) -> List[str]:
        constraints = []
        ptype = spec.problem_type.value if hasattr(spec, 'problem_type') else spec.get('problem_type', 'sorting')

        if ptype in ("sorting", "transformation"):
            if isinstance(data, (list, tuple)) and len(data) > 1:
                sample = data[:min(len(data), 1000)]
                try:
                    inversions = sum(
                        1 for i in range(len(sample) - 1)
                        for j in range(i + 1, min(len(sample), i + 50))
                        if sample[i] > sample[j]
                    )
                except TypeError:
                    inversions = len(sample)
                if inversions < len(sample) * 0.1:
                    constraints.append("NearlySorted")
                if len(data) < 10:
                    constraints.append("SmallN")
                elif len(data) > 10000:
                    constraints.append("LargeN")

        elif ptype == "pathfinding":
            graph = data if isinstance(data, dict) else {}
            if "graph" in data:
                graph = data["graph"]
            if isinstance(graph, dict):
                has_weights = any(
                    isinstance(v, dict) and any(isinstance(w, (int, float)) for w in v.values())
                    for v in graph.values()
                )
                constraints.append("WeightedGraph" if has_weights else "UnweightedGraph")

        elif ptype == "optimization":
            if "items" in (data if isinstance(data, dict) else {}):
                constraints.append("ResourceAllocation")

        return constraints

    def _build_state_vector(self, context: dict,
                            problem_spec: ProblemSpec = None) -> np.ndarray:
        vec = np.zeros(200, dtype=np.float32)
        dp = context.get("data_profile", {})
        patterns = dp.get("patterns", {})

        vec[0] = np.log10(dp.get("size", 1) + 1) / 5.0
        vec[1] = 1.0 if patterns.get("is_sorted") else 0.0
        vec[2] = 1.0 if patterns.get("is_nearly_sorted") else 0.0

        env = context.get("environment", {})
        cpu = env.get("cpu", {})
        mem = env.get("memory", {})
        vec[10] = (100.0 - cpu.get("percent_used", 50)) / 100.0
        vec[11] = mem.get("available_gb", 1) / max(mem.get("total_gb", 1), 1)

        if problem_spec:
            all_types = list(ProblemType)
            ptype = problem_spec.problem_type
            if ptype in all_types:
                idx = all_types.index(ptype)
                if 20 + idx < 200:
                    vec[20 + idx] = 1.0

        return vec

    def get_last_selection(self) -> Optional[Dict]:
        return self._last_selection

    def get_last_confidence(self) -> float:
        return self._last_confidence

    def _build_knowledge_graph(self):
        # Add ALL algorithm nodes FIRST, then link problem types to them
        algorithms = [
            ("quicksort", {"time_complexity": "O(n log n)", "patterns": ["DivideAndConquer", "ComparisonSort"], "best_for": ["RandomData", "LargeN"]}),
            ("timsort", {"time_complexity": "O(n log n)", "patterns": ["Hybrid", "Stable", "ComparisonSort"], "best_for": ["NearlySorted", "RealWorldData"]}),
            ("merge_sort", {"time_complexity": "O(n log n)", "patterns": ["DivideAndConquer", "Stable", "ComparisonSort"], "best_for": ["LargeN", "StabilityNeeded"]}),
            ("heap_sort", {"time_complexity": "O(n log n)", "patterns": ["ComparisonSort", "InPlace"], "best_for": ["MemoryConstrained", "LargeN"]}),
            ("insertion_sort", {"time_complexity": "O(n^2)", "patterns": ["ComparisonSort", "Incremental"], "best_for": ["SmallN", "NearlySorted"]}),
            ("radix_sort", {"time_complexity": "O(nk)", "patterns": ["NonComparisonSort", "DistributionBased"], "best_for": ["Integers", "UniformDistribution"]}),
            ("dijkstra", {"time_complexity": "O((V+E) log V)", "patterns": ["Greedy", "ShortestPath"], "best_for": ["WeightedGraph", "NonNegativeWeights"]}),
            ("a_star", {"time_complexity": "O(E)", "patterns": ["Heuristic", "ShortestPath"], "best_for": ["GeoGraph", "GridBased"]}),
            ("bfs_path", {"time_complexity": "O(V+E)", "patterns": ["BreadthFirst", "ShortestPath"], "best_for": ["UnweightedGraph", "PathExistence"]}),
            ("greedy_knapsack", {"time_complexity": "O(n log n)", "patterns": ["Greedy", "Approximation"], "best_for": ["FastApproximation", "ResourceAllocation"]}),
            ("simulated_annealing", {"time_complexity": "O(iterations * n)", "patterns": ["Metaheuristic", "Combinatorial"], "best_for": ["ComplexLandscape", "FineTuning"]}),
        ]
        for name, meta in algorithms:
            self.kg.add_algorithm(name, meta)

        self.kg.add_problem_type("sorting", [
            "quicksort", "insertion_sort", "merge_sort",
            "timsort", "radix_sort", "heap_sort",
        ])
        self.kg.add_problem_type("pathfinding", [
            "dijkstra", "a_star", "bfs_path",
        ])
        self.kg.add_problem_type("optimization", [
            "greedy_knapsack", "simulated_annealing",
        ])


class MetaController:
    def __init__(self, algorithm_registry: Dict[str, Any], strategy: str = "hybrid",
                 config: Optional[Dict] = None):
        self.registry = algorithm_registry
        self.strategy = strategy
        self.config = config or {}
        self.history: List[Dict] = []
        self.knowledge_base: Dict = defaultdict(list)
        self._trained = False
        self._model = None
        self._feature_names = [
            "data_size_log", "is_numeric", "is_sorted", "is_nearly_sorted",
            "cpu_free", "mem_free_ratio", "cpu_count", "time_budget_norm",
            "priority_speed", "priority_accuracy"
        ]

        self._domain_map = {
            "sorting": ["quicksort", "insertion_sort", "merge_sort", "timsort", "radix_sort", "heap_sort"],
            "image_processing": ["gaussian_blur", "median_filter", "bilateral_filter", "sobel_edge", "clahe"],
            "ml": ["kmeans", "dbscan", "random_forest", "linear_regression"]
        }

        self._population = []
        self._generation = 0
        self._performance_cache: Dict[str, Dict] = {}

        bandit_config = self.config.get("bandit", {})
        self.bandit = UCB1Bandit(
            algorithm_names=list(self.registry.keys()),
            epsilon=bandit_config.get("epsilon_initial", 0.2),
            epsilon_decay=bandit_config.get("epsilon_decay", 0.99),
            epsilon_min=bandit_config.get("epsilon_min", 0.05)
        )

        self.llm = LLMAdapter(self.config.get("llm", {}))
        self._last_confidence = 0.0
        self._last_reason = ""

    def select(self, context: Dict[str, Any]) -> List[Any]:
        algorithms, confidence, reason = self.select_with_confidence(context)
        self._last_confidence = confidence
        self._last_reason = reason
        return algorithms

    def select_with_confidence(self, context: Dict[str, Any]) -> Tuple[List[Any], float, str]:
        domain_algos = self._get_domain_algorithms(context)
        domain_names = [a for a in domain_algos if a in self.registry]
        if not domain_names:
            return [list(self.registry.values())[0]], 0.0, "fallback"

        similar = self.knowledge.query_similar(context, top_k=10) if hasattr(self, 'knowledge') else []
        kb_confidence = self._compute_kb_confidence(similar)
        history_confidence = min(1.0, len(self.history) / 100.0)
        confidence = 0.7 * kb_confidence + 0.3 * history_confidence

        if confidence < self.llm.config.get("confidence_threshold", 0.4) and self.llm.enabled:
            llm_result = self.llm.analyze_context(context)
            llm_algo = llm_result.get("algorithm")
            if llm_algo and llm_algo in self.registry and llm_algo in domain_names:
                reason = f"LLM: {llm_result.get('reason', 'LLM selected')}"
                return [self.registry[llm_algo]], confidence, reason

        if self.strategy == "rule-based":
            algo = self._rule_based_select_single(context)
            if algo in self.registry:
                return [self.registry[algo]], confidence, "rule-based"
        elif self.strategy == "ml-based":
            if self._trained and len(self.history) >= 50:
                ml_algo = self._ml_select_single(context)
                if ml_algo and ml_algo in self.registry:
                    return [self.registry[ml_algo]], confidence, "ml-based"
            return self._rule_based_select_with_confidence(context, confidence)
        elif self.strategy == "genetic":
            return self._genetic_select_with_confidence(context, confidence)
        else:
            return self._hybrid_select_with_confidence(context, confidence, domain_names)

        return [self.registry[domain_names[0]]], 0.0, "fallback"

    def _compute_kb_confidence(self, similar_records: List[Dict]) -> float:
        if not similar_records or len(similar_records) < 3:
            return 0.2
        scores = [s.get("metrics", {}).get("score", 0) for s in similar_records if s.get("metrics")]
        if not scores:
            return 0.2
        return min(1.0, max(0.0, np.mean(scores)))

    def _rule_based_select_single(self, context: Dict) -> Optional[str]:
        features = context.get("features", {})
        data_profile = context.get("data_profile", {})
        constraints = context.get("constraints", {})

        data_size = data_profile.get("size", 0)
        is_nearly_sorted = data_profile.get("patterns", {}).get("is_nearly_sorted", False)
        is_sorted = data_profile.get("patterns", {}).get("is_sorted", False)
        data_type = data_profile.get("type", "")

        cpu_free = features.get("cpu_free", 0.5)
        mem_free = features.get("mem_free_ratio", 0.5)
        priority = constraints.get("priority", "balanced")
        time_budget = constraints.get("time_budget_ms", 500)

        stats = data_profile.get("statistics", {})
        domain_algos = self._get_domain_algorithms(context)

        if data_type in ("list", "tuple", "ndarray"):
            if data_size <= 10:
                return "insertion_sort" if "insertion_sort" in domain_algos else domain_algos[0]
            elif is_sorted:
                return "timsort" if "timsort" in domain_algos else domain_algos[0]
            elif is_nearly_sorted and data_size > 50:
                return "timsort" if "timsort" in domain_algos else domain_algos[0]
            elif data_size > 100000:
                if stats.get("is_uniform", False) or stats.get("unique_ratio", 1.0) < 0.1:
                    return "radix_sort" if "radix_sort" in domain_algos else "timsort"
                return "timsort" if "timsort" in domain_algos else domain_algos[0]
            elif mem_free < 0.15 and data_size > 10000:
                return "heap_sort" if "heap_sort" in domain_algos else domain_algos[0]
            elif time_budget < 50 and data_size > 1000:
                return "timsort" if "timsort" in domain_algos else domain_algos[0]
            else:
                return "timsort" if "timsort" in domain_algos else domain_algos[0]

        return domain_algos[0] if domain_algos else None

    def _ml_select_single(self, context: Dict) -> Optional[str]:
        if not self._trained or len(self.history) < 50:
            return None
        try:
            features = context.get("features", {})
            X = self._vectorize(features)
            if self._model is not None:
                prediction = self._model.predict([X])[0]
                domain_algos = self._get_domain_algorithms(context)
                if prediction in domain_algos and prediction in self.registry:
                    return prediction
        except Exception:
            pass
        return None

    def _rule_based_select_with_confidence(self, context: Dict, confidence: float) -> Tuple[List[Any], float, str]:
        algo = self._rule_based_select_single(context)
        if algo and algo in self.registry:
            if confidence > 0.6:
                return [self.registry[algo]], confidence, "rule-based (confident)"
            chosen = self.bandit.select([algo])
            return [self.registry[chosen]], confidence, "rule-based + bandit"
        return [list(self.registry.values())[0]], 0.0, "fallback"

    def _genetic_select_with_confidence(self, context: Dict, confidence: float) -> Tuple[List[Any], float, str]:
        if len(self.history) < 100:
            algo = self._rule_based_select_single(context)
            return [self.registry[algo]], confidence, "genetic (no history)"

        if not self._population:
            self._initialize_population(context)

        if len(self.history) % 50 == 0:
            self._evolve_population(context)

        best = self._select_best_individual(context)
        if best and best in self.registry:
            domain_algos = self._get_domain_algorithms(context)
            if best in domain_algos:
                return [self.registry[best]], confidence, f"genetic (gen {self._generation})"

        algo = self._rule_based_select_single(context)
        return [self.registry[algo]], confidence, "genetic fallback"

    def _hybrid_select_with_confidence(self, context: Dict, confidence: float,
                                        domain_names: List[str]) -> Tuple[List[Any], float, str]:
        similar = self._find_similar_contexts(context, top_k=10)

        if len(self.history) < 10:
            rule_algo = self._rule_based_select_single(context)
            if rule_algo and rule_algo in self.registry:
                return [self.registry[rule_algo]], confidence, "hybrid (cold start -> rule)"

        if similar and len(similar) >= 3:
            best_algo = self._select_from_history_weighted(similar, context)
            if best_algo and best_algo in self.registry:
                avg_similarity = np.mean([
                    self._compute_similarity(context, r.get("context", {}))
                    for r in similar[:5]
                ])
                explore_threshold = max(0.05, 0.3 - avg_similarity * 0.25)

                if confidence > 0.6 and random.random() > explore_threshold:
                    return [self.registry[best_algo]], confidence, f"hybrid (similarity={avg_similarity:.2f})"
                else:
                    return self._smart_explore_with_confidence(context, best_algo, domain_names, similar, confidence)

        rule_algo = self._rule_based_select_single(context)
        if self._trained and confidence > 0.4:
            ml_algo = self._ml_select_single(context)
            if ml_algo and ml_algo != rule_algo and ml_algo in self.registry:
                ml_score = self._get_average_score(ml_algo)
                rule_score = self._get_average_score(rule_algo)
                if ml_score > rule_score * 1.1:
                    return [self.registry[ml_algo]], confidence, f"hybrid (ml={ml_score:.2f} > rule={rule_score:.2f})"

        if len(self.history) >= 10:
            chosen = self.bandit.select(domain_names)
            return [self.registry[chosen]], confidence, "hybrid + bandit"

        return [self.registry[rule_algo]], confidence, "hybrid (history < 10)"

    def _compute_similarity(self, context_a: Dict, context_b: Dict) -> float:
        features_a = context_a.get("features", {})
        features_b = context_b.get("features", {})

        vec_a = np.array(self._vectorize(features_a))
        vec_b = np.array(self._vectorize(features_b))

        dot = np.dot(vec_a, vec_b)
        norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

        if norms == 0:
            return 0.0
        return float(dot / norms)

    def _find_similar_contexts(self, context: Dict, top_k: int = 5) -> List[Dict]:
        if not self.history:
            return []

        similarities = []
        for record in self.history:
            record_context = record.get("context", {})
            similarity = self._compute_similarity(context, record_context)
            similarities.append((similarity, record))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in similarities[:top_k]]

    def _select_from_history_weighted(self, similar_records: List[Dict], current_context: Dict) -> Optional[str]:
        algo_scores = defaultdict(list)
        algo_similarities = defaultdict(list)

        for record in similar_records:
            algos = record.get("algorithms", [])
            score = record.get("score", 0)
            similarity = self._compute_similarity(current_context, record.get("context", {}))

            for algo in algos:
                algo_scores[algo].append(score * similarity)
                algo_similarities[algo].append(similarity)

        if not algo_scores:
            return None

        def weighted_avg(algo):
            scores = algo_scores[algo]
            sims = algo_similarities[algo]
            total_weight = sum(sims)
            if total_weight == 0:
                return 0
            return sum(scores) / total_weight

        best_algo = max(algo_scores.keys(), key=weighted_avg)
        return best_algo

    def _smart_explore_with_confidence(self, context: Dict, current_best: str,
                                        domain_algos: List[str], similar_records: List[Dict],
                                        confidence: float) -> Tuple[List[Any], float, str]:
        tried_algos = set()
        for record in similar_records:
            tried_algos.update(record.get("algorithms", []))

        untried = [a for a in domain_algos if a not in tried_algos and a in self.registry]

        if untried:
            chosen = self.bandit.select(untried)
            return [self.registry[chosen]], confidence, f"exploring untried ({chosen})"

        current_best_score = self._get_average_score(current_best)
        alternatives = []
        for algo in domain_algos:
            if algo != current_best and algo in self.registry:
                algo_score = self._get_average_score(algo)
                if algo_score > current_best_score * 0.5:
                    alternatives.append(algo)

        if alternatives:
            chosen = self.bandit.select(alternatives)
            return [self.registry[chosen]], confidence, f"exploring alternative ({chosen})"

        return [self.registry[current_best]], confidence, "sticking with best"

    def get_last_confidence(self) -> float:
        return self._last_confidence

    def get_last_reason(self) -> str:
        return self._last_reason

    def execute_with_fallback(self, data: Any, context: Dict, pipeline_algorithms: List[Any]) -> Tuple[Any, bool]:
        try:
            result = data
            for algo in pipeline_algorithms:
                result = algo.process(result)
            return result, True
        except Exception:
            safe_name = self._rule_based_select_single(context)
            if safe_name and safe_name in self.registry:
                return self.registry[safe_name].process(data), True
            return data, False

    def _get_domain_algorithms(self, context: Dict) -> List[str]:
        task_type = context.get("task_type", "sorting")
        domain = task_type
        if task_type in ("auto", ""):
            data_type = context.get("data_profile", {}).get("type", "")
            if data_type in ("list", "tuple"):
                domain = "sorting"
            elif data_type == "ndarray":
                domain = "image_processing"
            else:
                domain = "sorting"
        return self._domain_map.get(domain, list(self.registry.keys()))

    def _get_average_score(self, algo_name: str) -> float:
        scores = [
            r.get("score", 0)
            for r in self.history
            if algo_name in r.get("algorithms", [])
        ]
        return np.mean(scores) if scores else 0.5

    def _vectorize(self, features: Dict) -> List[float]:
        return [features.get(f, 0.0) for f in self._feature_names]

    def _initialize_population(self, context: Dict):
        domain_algos = self._get_domain_algorithms(context)
        self._population = [
            random.choice(domain_algos)
            for _ in range(20)
        ]

    def _evolve_population(self, context: Dict):
        if not self.history:
            return

        domain_algos = self._get_domain_algorithms(context)

        fitness = {}
        for individual in set(self._population):
            if individual not in domain_algos:
                fitness[individual] = 0.0
                continue
            scores = [
                r.get("score", 0)
                for r in self.history
                if individual in r.get("algorithms", [])
            ]
            fitness[individual] = np.mean(scores) if scores else 0.0

        new_population = []
        for _ in range(len(self._population)):
            candidates = random.sample(self._population, min(3, len(self._population)))
            winner = max(candidates, key=lambda x: fitness.get(x, 0))
            new_population.append(winner)

        for i in range(0, len(new_population) - 1, 2):
            if random.random() < 0.7:
                alternatives = [a for a in domain_algos if a != new_population[i]]
                if alternatives:
                    new_population[i] = random.choice(alternatives)

        self._population = new_population
        self._generation += 1

    def _select_best_individual(self, context: Dict) -> Optional[str]:
        if not self._population:
            return None

        similar = self._find_similar_contexts(context, top_k=10)
        if similar:
            return self._select_from_history_weighted(similar, context)

        domain_algos = self._get_domain_algorithms(context)
        valid_pop = [p for p in self._population if p in domain_algos]
        if valid_pop:
            return max(set(valid_pop), key=lambda x: self._get_average_score(x))
        return None

    def train(self):
        if len(self.history) < 50:
            return

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import StratifiedKFold
            from collections import Counter

            X = []
            y = []

            for record in self.history:
                ctx = record.get("context", {})
                features = ctx.get("features", {})
                algos = record.get("algorithms", [])

                if algos:
                    X.append(self._vectorize(features))
                    y.append(algos[0])

            if len(set(y)) < 2:
                return

            class_counts = Counter(y)
            min_class_count = min(class_counts.values())

            if min_class_count < 2:
                return

            n_splits = min(3, min_class_count)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            model = RandomForestClassifier(
                n_estimators=min(50, len(X) // 2),
                max_depth=5,
                min_samples_split=2,
                random_state=42
            )

            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=cv)
            except Exception:
                pass

            model.fit(X, y)
            self._model = model
            self._trained = True

        except ImportError:
            pass
        except Exception:
            pass

    def record(self, context: Dict, algorithms: List[Any], score: float, metrics: Dict):
        algo_names = [a.name if hasattr(a, "name") else str(a) for a in algorithms]

        record = {
            "context": context,
            "algorithms": algo_names,
            "score": score,
            "metrics": metrics,
            "timestamp": time.time()
        }

        self.history.append(record)

        for algo_name in algo_names:
            self.knowledge_base[algo_name].append({
                "context_features": context.get("features", {}),
                "score": score,
                "metrics": metrics
            })

        if len(self.history) % 100 == 0 and self.strategy in ("ml-based", "hybrid"):
            self.train()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "history_size": len(self.history),
            "trained": self._trained,
            "generation": self._generation,
            "last_confidence": self._last_confidence,
            "last_reason": self._last_reason,
            "bandit": self.bandit.get_stats(),
            "llm": self.llm.get_stats(),
            "algorithm_performance": {
                name: {
                    "avg_score": np.mean([r["score"] for r in records]) if records else 0,
                    "count": len(records)
                }
                for name, records in self.knowledge_base.items()
            }
        }
