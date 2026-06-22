import logging
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from typing import Any

import numpy as np

from aalgoi.core.bandit import UCB1Bandit
from aalgoi.core.knowledge_graph import AlgorithmKnowledgeGraph
from aalgoi.core.llm_adapter import LLMAdapter
from aalgoi.core.problem_spec import ProblemSpec, ProblemType

logger = logging.getLogger(__name__)


class UniversalMetaController:
    def __init__(self, config: dict = None, knowledge_base: Any = None,
                 algorithm_registry: dict = None,
                 problem_library: Any = None, llm_client: Any = None,
                 mode: str = "standard") -> None:
        self.config = config or {}
        self.kb = knowledge_base
        self.registry = algorithm_registry or {}
        self.history = []
        self._last_selection = None
        self._last_confidence = 0.0
        self._cross_domain_pool = []
        self._last_state = None
        self._last_action_idx = None
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_calibrated_entropy = 0.0

        self._mode = mode

        if self._mode == "universal":
            from aalgoi.core.mind import create_mind
            from aalgoi.core.mind.model_config import DEFAULT_CONFIG
            self._mind_loop = create_mind(
                config=DEFAULT_CONFIG,
                solver=getattr(self, '_solver', None),
            )
            return

        # Configurable features (off by default for minimal MVP)
        self.use_bandit = self.config.get('use_bandit', False)
        self.use_drift = self.config.get('use_drift', False)
        self.use_genetic = self.config.get('use_genetic', False)

        # Lazy initialization — created on first use
        self._rl_agent_init = False
        self._rl_agent = None
        self._rl_episode_buffer = []

        self.fallback_chain = self._build_fallback_chain()

        # Domain routing + per-domain bandits for hybrid PPO + bandit selection
        self._domain_bandits: dict[str, UCB1Bandit] = {}
        self.bandit_weight = self.config.get("bandit_weight", 0.3)
        from aalgoi.core.rl.domain_router import _build_domain_map
        _build_domain_map(self.registry)

        # Knowledge Graph — lazy
        self._kg_initialized = False
        self._kg = None
        self.kg_enabled = self.config.get('kg_enabled', True)

        # Pre-trained weights — lazy (loaded with RL agent)
        self._pretrained_loaded = False

        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.6)
        self.latency_sla_ms = self.config.get("latency_sla_ms", 200)
        self.timouts = {
            "rl_agent":    self.config.get("timeout_rl_ms", 50) / 1000.0,
            "kg_traversal": self.config.get("timeout_kg_ms", 100) / 1000.0,
            "chromadb":    self.config.get("timeout_chromadb_ms", 100) / 1000.0,
            "llm":         self.config.get("timeout_llm_s", 5.0),
        }

        logger.info(
            f"UniversalMetaController initialized: "
            f"{len(self.registry)} algorithms (lazy RL/KG)"
        )

    @property
    def rl_agent(self) -> Any:
        if not self._rl_agent_init:
            self._init_rl()
            self._rl_agent_init = True
        return self._rl_agent

    @rl_agent.setter
    def rl_agent(self, value: Any) -> None:
        self._rl_agent = value
        self._rl_agent_init = True

    @property
    def kg(self) -> Any:
        if not self._kg_initialized:
            self._kg = AlgorithmKnowledgeGraph()
            self._build_knowledge_graph()
            self._kg_initialized = True
        return self._kg

    @kg.setter
    def kg(self, value: Any) -> None:
        self._kg = value
        self._kg_initialized = True

    def _init_rl(self) -> None:
        try:
            from aalgoi.core.rl.agents.selection_agent import PPOAgent
            rl_config = self.config.get("rl", {}).get("agents", {}).get("selection", {})
            self._rl_agent = PPOAgent(
                state_dim=42,
                config=rl_config,
            )
            self._rl_episode_buffer = []
            logger.info("RL agent created: state_dim=42 (attention head)")
            self._load_pretrained_model()
        except Exception as e:
            logger.warning(f"Failed to init RL agent: {e}")

    def _load_pretrained_model(self) -> None:
        agent = self._rl_agent
        if agent is None:
            logger.info("No RL agent available, skipping pretrained model load")
            return

        from aalgoi.core.checkpoint_downloader import (
            checkpoint_exists,
            get_checkpoint_path,
        )

        local_candidates = [
            os.path.expanduser("~/.aalgoi/checkpoints/pretrained_final.pt"),
            os.path.expanduser("~/.aalgoi/checkpoints/pretrained_v1.pt"),
        ]

        for path in local_candidates:
            if os.path.exists(path) and agent.load(path):
                logger.info(f"Loaded pre-trained RL weights: {path}")
                return

        if not checkpoint_exists():
            return

        checkpoint_path = get_checkpoint_path()
        if os.path.exists(checkpoint_path) and agent.load(checkpoint_path):
            logger.info(f"Loaded pre-trained RL weights: {checkpoint_path}")
            return

        logger.info("No pretrained model found (fresh RL agent)")

    @staticmethod
    def _with_timeout(func: Any, timeout_s: float, default: Any, *args: Any, **kwargs: Any) -> Any:
        result_holder = [default]
        exc_holder = [None]

        def wrapper() -> None:
            try:
                result_holder[0] = func(*args, **kwargs)
            except Exception as e:
                exc_holder[0] = e

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            logger.warning("Timeout (%ss) exceeded for %s", timeout_s, getattr(func, '__name__', 'callable'))
            return default
        if exc_holder[0] is not None:
            logger.warning("Error in timed call %s: %s", getattr(func, '__name__', 'callable'), exc_holder[0])
            return default
        return result_holder[0]

    def select(self, context: dict, candidates: list = None,
               problem_spec: ProblemSpec = None, query: str = "") -> Any:
        if self._mode == "universal" and hasattr(self, '_mind_loop'):
            solution = self._mind_loop.solve(
                getattr(problem_spec, 'description', query or ""),
                context.get("data", {}),
                examples=getattr(problem_spec, 'examples', None),
            )
            return solution.code, solution.output, solution.principle_applied

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
        if not query:
            query = problem_spec.description or problem_spec.name or ""

        # 1. KG narrows candidates by problem type + constraints
        kg_pool = None
        self._cross_domain_pool = []
        if self.kg_enabled and problem_spec:
            ptype_val = problem_spec.problem_type.value if hasattr(problem_spec, 'problem_type') else ""
            constraints = self._extract_kg_constraints(problem_spec, context.get("data", {}))
            semantic_candidates = self.kg.find_candidates(ptype_val, constraints)

            # Find cross-domain candidates as fallback when primary pool is small
            if len(semantic_candidates) < 3:
                self._cross_domain_pool = self._with_timeout(
                    self.kg.find_cross_domain_candidates,
                    self.timouts["kg_traversal"],
                    [],
                    ptype_val, constraints, max_hops=4,
                )

            if semantic_candidates:
                {c if isinstance(c, str) else c.get("name", "") for c in candidates}
                filtered = [c for c in candidates
                            if (c if isinstance(c, str) else c.get("name", "")) in semantic_candidates]
                if filtered:
                    kg_pool = filtered
                    candidates = filtered
            elif self._cross_domain_pool:
                # No primary candidates at all — use cross-domain
                {c if isinstance(c, str) else c.get("name", "") for c in candidates}
                filtered = [c for c in candidates
                            if (c if isinstance(c, str) else c.get("name", "")) in self._cross_domain_pool]
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

        # 2. RL picks (skip if untrained — buffer below minimum)
        self._synthesized_algo = None
        rl_ready = (self.rl_agent is not None
                    and hasattr(self.rl_agent, 'buffer')
                    and len(self.rl_agent.buffer) >= 100)
        if rl_ready:
            use_deterministic = self.config.get("rl_deterministic", True)

            # Build candidate mask from kg_pool (restricts RL action space)
            state = self._build_state_vector(context, problem_spec)
            algo_names = list(self.registry.keys())
            candidate_mask = None
            pool_names = set()
            if kg_pool:
                pool_names = {c if isinstance(c, str) else c.get("name", "") for c in kg_pool}
                candidate_mask = [
                    i for i, n in enumerate(algo_names) if n in pool_names
                ]

            # Domain-specific routing — restrict to algorithms matching problem type
            if problem_spec and problem_spec.problem_type != ProblemType.UNKNOWN:
                from aalgoi.core.rl.domain_router import build_candidate_mask as _domain_mask
                domain_mask = _domain_mask(algo_names, problem_spec.problem_type)
                if domain_mask:
                    pool_names = {algo_names[i] for i in domain_mask}
                    if candidate_mask is not None:
                        candidate_mask = [i for i in candidate_mask if i in domain_mask]
                    else:
                        candidate_mask = domain_mask

            rl_result = self._with_timeout(
                lambda: self.rl_agent.select_action(
                    state, deterministic=use_deterministic, candidate_mask=candidate_mask,
                ),
                self.timouts["rl_agent"],
                (0, 0.0, 0.0),
            )
            if rl_result == (0, 0.0, 0.0) and not self._rl_agent_init:
                action_idx, log_prob, value = 0, 0.0, 0.0
            else:
                action_idx, log_prob, value = rl_result
            raw_name = algo_names[action_idx % len(algo_names)]

            if candidate_mask is not None and raw_name not in pool_names:
                best_kg = self._select_best_algorithm(list(pool_names), query, context) if pool_names else None
                algo_name = best_kg if best_kg and best_kg in self.registry else (list(pool_names)[0] if pool_names else raw_name)
            else:
                algo_name = raw_name

            # Hybrid: combine PPO probs with per-domain bandit scores
            if self.use_bandit and problem_spec:
                domain_key = problem_spec.problem_type.value
                if domain_key not in self._domain_bandits:
                    domain_algos = [n for n in algo_names if n in pool_names] if pool_names else algo_names
                    self._domain_bandits[domain_key] = UCB1Bandit(
                        algorithm_names=domain_algos,
                        epsilon=0.15,
                    )
                domain_bandit = self._domain_bandits[domain_key]
                try:
                    cal_probs, _ = self.rl_agent.get_calibrated_probs(state)
                    if candidate_mask:
                        cal_probs_np = cal_probs
                        bandit_scores = np.array([
                            domain_bandit.rewards.get(n, 0.0) / max(domain_bandit.counts.get(n, 1), 1)
                            for n in algo_names
                        ])
                        bandit_scores = np.where(bandit_scores == 0, 1e-6, bandit_scores)
                        bandit_probs = np.exp(bandit_scores - bandit_scores.max())
                        bandit_probs = bandit_probs / bandit_probs.sum()
                        combined = (1.0 - self.bandit_weight) * cal_probs_np + self.bandit_weight * bandit_probs
                        combined_idx = int(np.argmax(combined))
                        algo_name = algo_names[combined_idx % len(algo_names)]
                except Exception:
                    pass

            self._last_confidence = min(1.0, max(0.0, float(value)))
            self._last_state = state
            self._last_action_idx = action_idx
            self._last_log_prob = log_prob
            self._last_value = value

            # Calibrated uncertainty — escalate on high entropy
            try:
                cal_probs, cal_entropy = self.rl_agent.get_calibrated_probs(state)
                self._last_calibrated_entropy = cal_entropy
            except Exception:
                self._last_calibrated_entropy = 0.0

            # Fast path: high confidence → return immediately without fallback layers
            if self._last_confidence >= self.uncertainty_threshold:
                self._last_selection = {"algorithm": algo_name, "confidence": self._last_confidence}
                return self.registry[algo_name]

            if self._last_confidence < 0.5 and query and kg_pool:
                scored_pick = self._select_best_algorithm(kg_pool, query, context)
                if scored_pick and scored_pick in self.registry:
                    rl_score = self._score_algorithm_for_query(algo_name, query) if algo_name in self.registry else 0.0
                    best_score = self._score_algorithm_for_query(scored_pick, query)
                    if best_score > rl_score + 0.05:
                        algo_name = scored_pick
                        logger.info(f"RL override: {scored_pick} (score={best_score:.3f}) > {rl_score:.3f}")

            if algo_name in self.registry:
                # Attempt LLM synthesis when confidence is low
                if (self._last_confidence < 0.4
                        and self.config.get('llm_synthesis_enabled', False)
                        and problem_spec is not None):
                    try:
                        from aalgoi.core.algorithm_synthesizer import LLMAlgorithmSynthesizer
                        data = context.get("data", {})
                        baseline = self.registry[algo_name]
                        synth = LLMAlgorithmSynthesizer()
                        syn_algo = synth.synthesize(problem_spec, data, baseline)
                        if syn_algo is not None:
                            self._synthesized_algo = syn_algo.name
                            # Register it
                            self.registry[syn_algo.name] = syn_algo
                            # Update KG
                            if self.kg_enabled and hasattr(self, '_kg') and self._kg is not None:
                                self.kg.add_algorithm(syn_algo.name, {
                                    "time_complexity": syn_algo.time_complexity,
                                    "best_for": syn_algo.best_for,
                                })
                                self.kg.add_problem_type(
                                    problem_spec.problem_type.value,
                                    [syn_algo.name],
                                )
                            logger.info(f"LLM synthesized new algorithm: {syn_algo.name}")
                            return syn_algo
                    except Exception as exc:
                        logger.warning(f"LLM synthesis failed: {exc}", exc_info=True)

                self._last_selection = {"algorithm": algo_name, "confidence": self._last_confidence}
                return self.registry[algo_name]

        # 3. Scoring-based selection: pick best candidate by query metadata
        pick = self._select_best_algorithm(candidates, query, context)
        if pick and pick in self.registry:
            return self.registry[pick]

        fallbacks = self._get_fallback_algorithms()
        if fallbacks:
            pick = self._select_best_algorithm(
                [fb.name for fb in fallbacks], query, context
            )
            if pick and pick in self.registry:
                return self.registry[pick]
            return fallbacks[0]
        if self.registry:
            return list(self.registry.values())[0]
        return None

    def get_fallback_alternative(self, failed_algo_name: str) -> str | None:
        if not self.kg_enabled:
            return None
        alternatives = self.kg.find_alternatives(failed_algo_name)
        if alternatives:
            return alternatives[0]
        if self._cross_domain_pool:
            return self._cross_domain_pool[0]
        return None

    def _build_fallback_chain(self) -> list[str]:
        """Ordered list of safe algorithms to try when all else fails."""
        return [
            "identity",
            "safe_sort",
            "quicksort",
            "bfs_path",
            "greedy_knapsack",
        ]

    def _get_fallback_algorithms(self) -> list[Any]:
        """Return fallback algorithms from the chain that exist in registry."""
        fallbacks = []
        for name in self.fallback_chain:
            if name in self.registry:
                fallbacks.append(self.registry[name])
        return fallbacks

    @staticmethod
    def _split_words(text: str) -> set:
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return set(re.findall(r'[a-z0-9]+', s.lower()))

    @staticmethod
    def _action_words() -> dict[str, set[str]]:
        return {
            "sort": {"sort", "sorting", "rank", "order", "arrange"},
            "path": {"path", "route", "shortest", "find path", "traverse"},
            "classify": {"classify", "classification", "label", "predict class", "tier"},
            "regress": {"regress", "regression", "predict value", "forecast", "growth"},
            "cluster": {"cluster", "group", "segment", "partition"},
            "reduce": {"reduce", "pca", "tsne", "t-sne", "dimensionality", "compress"},
            "sentiment": {"sentiment", "opinion", "emotion", "feeling"},
            "summarize": {"summarize", "summary", "condense", "shorten"},
            "retrieve": {"retrieve", "rag", "passage", "find in document"},
            "search": {"search", "find similar", "match document"},
            "enrich": {"enrich", "enhance", "improve prompt", "expand prompt"},
            "arithmetic": {"arithmetic", "analogy", "minus", "plus", "king queen"},
            "visualize": {"visualize", "visualization", "plot", "2d", "3d", "coordinates"},
            "expand": {"expand", "expansion", "related terms", "broader"},
            "generate": {"generate", "creative", "write", "sentence", "story"},
            "train": {"train", "training", "learn embeddings", "fit"},
            "edge": {"edge", "detect edges", "gradient", "canny", "sobel"},
            "segment": {"segment", "segmentation", "region"},
            "template": {"template", "match pattern", "find pattern"},
            "morphological": {"morphological", "morph", "dilate", "erode", "close", "open"},
            "knapsack": {"knapsack", "capacity", "maximize value", "budget"},
            "annealing": {"annealing", "simulated annealing", "temperature"},
            "genetic": {"genetic", "evolution", "crossover", "mutation", "population"},
            "hill": {"hill climbing", "hill", "climb", "local search"},
            "pso": {"particle", "swarm", "pso"},
            "aco": {"ant colony", "aco", "pheromone", "ant"},
            "greedy": {"greedy", "heuristic", "best first"},
        }

    def _score_algorithm_for_query(self, algo_name: str, query: str) -> float:
        algo = self.registry.get(algo_name)
        if not algo:
            return 0.0
        meta = algo.metadata() if hasattr(algo, "metadata") else {}
        query_lower = query.lower()
        query_words = set(query_lower.replace("-", " ").replace("_", " ").split())
        score = 0.0

        # 1. Name-part match — strongest signal (weight 15)
        name_parts = set(algo_name.replace("_", " ").split())
        name_hits = 0
        for part in name_parts:
            if len(part) > 2 and part in query_lower:
                name_hits += 1
        score += name_hits * 15.0
        if name_hits == 0:
            score -= 5.0

        # 2. Tag match (weight 5 exact, 2 partial)
        for tag in meta.get("tags", []):
            tag_lower = tag.lower()
            if tag_lower in query_lower:
                score += 5.0
            tag_words = set(tag_lower.replace("-", " ").replace("_", " ").split())
            overlap = query_words & tag_words
            score += len(overlap) * 2.0

        # 3. Best-for match (weight 8 — very discriminative)
        for bf in meta.get("best_for", []):
            bf_words = set(bf.lower().replace("_", " ").replace("-", " ").split())
            overlap = query_words & bf_words
            score += len(overlap) * 8.0

        # 4. Pattern match (weight 3)
        for pat in meta.get("patterns", []):
            if pat.lower() in query_lower:
                score += 3.0

        # 5. Action-based penalty — if query clearly wants one action
        # and algorithm doesn't support it
        action_words = self._action_words()
        query_actions = set()
        for action, keywords in action_words.items():
            if any(kw in query_lower for kw in keywords):
                query_actions.add(action)
        algo_text = (algo_name + " " +
                     " ".join(meta.get("tags", [])) + " " +
                     " ".join(meta.get("patterns", []))).lower()
        algo_actions = set()
        for action, keywords in action_words.items():
            if any(kw in algo_text for kw in keywords):
                algo_actions.add(action)
        if query_actions and algo_actions:
            overlap = query_actions & algo_actions
            if not overlap:
                score -= 20.0
        return score

    def _select_best_algorithm(self, candidates: list, query: str = "",
                               context: dict = None) -> str | None:
        if not candidates:
            return None
        names = []
        for c in candidates:
            if isinstance(c, str):
                names.append(c)
            elif isinstance(c, dict):
                names.append(c.get("name", ""))
            else:
                n = getattr(c, "name", str(c))
                names.append(n)
        names = [n for n in names if n in self.registry]
        if not names:
            return None
        if not query and context:
            query = context.get("query", "")
        if not query:
            return names[0]
        scored = [(n, self._score_algorithm_for_query(n, query)) for n in names]
        scored.sort(key=lambda x: (-x[1], names.index(x[0])))
        return scored[0][0]

    def _extract_kg_constraints(self, spec: ProblemSpec, data: Any) -> list[str]:
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
            if "start" in (data if isinstance(data, dict) else {}):
                constraints.append("PointToPoint")

        elif ptype == "optimization":
            if "items" in (data if isinstance(data, dict) else {}):
                constraints.append("ResourceAllocation")

        elif ptype in ("classification", "regression", "clustering", "ml"):
            X = data.get("X_train") if isinstance(data, dict) else data
            if hasattr(X, 'shape') and len(X.shape) == 2:
                n, d = X.shape
                if d > 100:
                    constraints.append("HighDim")
                if d > 1000:
                    constraints.append("VeryHighDim")
                if n < 100:
                    constraints.append("SmallN")
                if n > 10000:
                    constraints.append("LargeN")
                if n > 100000:
                    constraints.append("VeryLargeN")
                try:
                    from scipy import sparse
                    if sparse.issparse(X):
                        sparsity = 1.0 - (X.nnz / (n * d))
                        if sparsity > 0.5:
                            constraints.append("SparseData")
                except ImportError:
                    pass

        return constraints

    def _build_state_vector(self, context: dict,
                            problem_spec: ProblemSpec = None) -> np.ndarray:
        vec = np.zeros(42, dtype=np.float32)
        profile = context.get('data_profile', {})
        stats   = profile.get('statistics', {})
        patterns = profile.get('patterns', {})
        env     = context.get('environment', {})
        cpu     = env.get('cpu', {})
        memory  = env.get('memory', {})

        size = profile.get('size', 0)

        vec[0] = np.log10(size + 1) / 7.0
        vec[1] = 1.0 if patterns.get('is_sorted') else 0.0
        vec[2] = 1.0 if patterns.get('is_nearly_sorted') else 0.0
        vec[3] = 1.0 if patterns.get('is_reverse_sorted') else 0.0
        vec[4] = float(stats.get('unique_ratio', 0.0))
        vec[5] = 1.0 if stats.get('unique_ratio', 1.0) < 1.0 else 0.0
        vec[6] = 1.0 if stats.get('min', 0) < 0 else 0.0
        vec[7] = 1.0 if profile.get('has_nulls', False) else 0.0

        mean  = stats.get('mean', 0.0)
        std   = stats.get('std', 1.0)
        vec[8]  = float(np.tanh(mean / (std + 1e-8)))
        vec[9]  = float(min(1.0, std / (abs(mean) + 1e-8)))
        vec[10] = float(np.tanh(stats.get('skewness', 0.0)))
        vec[11] = float(np.tanh(stats.get('kurtosis', 0.0) / 10.0))
        vec[12] = float(np.log10(abs(stats.get('max', 1) - stats.get('min', 0)) + 1) / 10.0)
        vec[13] = float(stats.get('sparsity_ratio', 0.0))

        vec[14] = 1.0 - cpu.get('percent_used', 0.0) / 100.0
        vec[15] = memory.get('available_gb', 1.0) / max(memory.get('total_gb', 1.0), 1.0)
        vec[16] = 1.0 if 'torch' in sys.modules and getattr(sys.modules['torch'], 'cuda', None) and sys.modules['torch'].cuda.is_available() else 0.0
        vec[17] = min(1.0, context.get('constraints', {}).get('time_budget_ms', 500) / 1000.0)

        TYPE_ORDER = [
            ProblemType.SORTING, ProblemType.PATHFINDING,
            ProblemType.OPTIMIZATION, ProblemType.CLUSTERING,
            ProblemType.CLASSIFICATION, ProblemType.REGRESSION,
            ProblemType.ML, ProblemType.IMAGE_PROCESSING,
            ProblemType.SEARCH, ProblemType.NLP,
            ProblemType.ROUTING, ProblemType.TRANSFORMATION,
            ProblemType.GENERATION, ProblemType.DECISION,
            ProblemType.SCHEDULING, ProblemType.COMPUTER_VISION,
        ]
        if problem_spec and problem_spec.problem_type in TYPE_ORDER:
            vec[18 + TYPE_ORDER.index(problem_spec.problem_type)] = 1.0

        constraints_map = context.get('constraints', {})
        vec[34] = 1.0 if constraints_map.get('requires_stable_sort', False) else 0.0
        vec[35] = 1.0 if constraints_map.get('requires_optimal', False) else 0.0
        vec[36] = 1.0 if constraints_map.get('memory_constrained', False) else 0.0
        vec[37] = 1.0 if constraints_map.get('time_constrained', False) else 0.0
        vec[38] = 1.0 if constraints_map.get('requires_parallel', False) else 0.0
        vec[39] = 1.0 if constraints_map.get('is_streaming', False) else 0.0
        vec[40] = 1.0 if size > 100_000 else 0.0
        vec[41] = 1.0 if size < 100 else 0.0

        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)

        return vec

    def get_last_selection(self) -> dict | None:
        return self._last_selection

    def get_last_confidence(self) -> float:
        return self._last_confidence

    def _build_knowledge_graph(self) -> None:
        kg = self._kg
        algorithms = [
            ("quicksort", {"time_complexity": "O(n log n)", "patterns": ["DivideAndConquer", "ComparisonSort"], "best_for": ["RandomData", "LargeN"]}),
            ("timsort", {"time_complexity": "O(n log n)", "patterns": ["Hybrid", "Stable", "ComparisonSort"], "best_for": ["NearlySorted", "RealWorldData"]}),
            ("merge_sort", {"time_complexity": "O(n log n)", "patterns": ["DivideAndConquer", "Stable", "ComparisonSort"], "best_for": ["LargeN", "StabilityNeeded"]}),
            ("heap_sort", {"time_complexity": "O(n log n)", "patterns": ["ComparisonSort", "InPlace"], "best_for": ["MemoryConstrained", "LargeN"]}),
            ("insertion_sort", {"time_complexity": "O(n^2)", "patterns": ["ComparisonSort", "Incremental"], "best_for": ["SmallN", "NearlySorted"]}),
            ("radix_sort", {"time_complexity": "O(nk)", "patterns": ["NonComparisonSort", "DistributionBased"], "best_for": ["Integers", "UniformDistribution"]}),
            ("dijkstra", {"time_complexity": "O((V+E) log V)", "patterns": ["Greedy", "ShortestPath"], "best_for": ["WeightedGraph", "NonNegativeWeights", "PointToPoint"]}),
            ("a_star", {"time_complexity": "O(E)", "patterns": ["Heuristic", "ShortestPath"], "best_for": ["GeoGraph", "GridBased", "PointToPoint"]}),
            ("bfs_path", {"time_complexity": "O(V+E)", "patterns": ["BreadthFirst", "ShortestPath"], "best_for": ["UnweightedGraph", "PathExistence", "PointToPoint"]}),
            ("greedy_knapsack", {"time_complexity": "O(n log n)", "patterns": ["Greedy", "Approximation"], "best_for": ["FastApproximation", "ResourceAllocation"]}),
            ("simulated_annealing", {"time_complexity": "O(iterations * n)", "patterns": ["Metaheuristic", "Combinatorial"], "best_for": ["ComplexLandscape", "FineTuning"]}),
        ]
        for name, meta in algorithms:
            kg.add_algorithm(name, meta)

        ml_algorithms = [
            ("linear_regression", {"time_complexity": "O(n*d\u00b2)", "patterns": ["LinearModel", "Parametric"], "best_for": ["SmallN", "LinearRelationship", "Baseline"]}),
            ("ridge", {"time_complexity": "O(n*d\u00b2)", "patterns": ["LinearModel", "Regularized"], "best_for": ["Multicollinearity", "OverfittingPrevention"]}),
            ("lasso", {"time_complexity": "O(n*d\u00b2)", "patterns": ["LinearModel", "Regularized", "FeatureSelection"], "best_for": ["SparseSolutions", "FeatureSelection"]}),
            ("logistic_regression", {"time_complexity": "O(n*d)", "patterns": ["LinearModel", "Probabilistic"], "best_for": ["BinaryClassification", "ProbabilityOutput"]}),
            ("knn", {"time_complexity": "O(n*d)", "patterns": ["DistanceBased", "NonParametric"], "best_for": ["SmallN", "DecisionBoundary"]}),
            ("svm", {"time_complexity": "O(n\u00b2*d)", "patterns": ["KernelMethod", "MarginBased"], "best_for": ["HighDim", "ClearMargin", "SmallMediumN"]}),
            ("gaussian_nb", {"time_complexity": "O(n*d)", "patterns": ["Probabilistic", "NaiveBayes"], "best_for": ["HighDim", "RealTime", "SmallN"]}),
            ("random_forest_classification", {"time_complexity": "O(n*m*log(n))", "patterns": ["Ensemble", "TreeBased", "Bagging"], "best_for": ["Tabular", "Nonlinear", "MixedFeatures"]}),
            ("xgboost_classification", {"time_complexity": "O(n*m*log(n))", "patterns": ["Ensemble", "Boosting", "GradientBoosting"], "best_for": ["Tabular", "AccuracyCritical", "Competitions"]}),
            ("lightgbm_classification", {"time_complexity": "O(n*m*log(n))", "patterns": ["Ensemble", "Boosting", "GradientBoosting"], "best_for": ["LargeData", "SpeedCritical", "Tabular"]}),
            ("kmeans", {"time_complexity": "O(n*k*i)", "patterns": ["CentroidBased", "Partitioning"], "best_for": ["SphericalClusters", "LargeN", "Fast"]}),
            ("dbscan", {"time_complexity": "O(n*log(n))", "patterns": ["DensityBased", "NoiseDetection"], "best_for": ["ArbitraryShapes", "NoiseDetection"]}),
            ("gmm", {"time_complexity": "O(n*k*d)", "patterns": ["Probabilistic", "SoftClustering"], "best_for": ["DensityEstimation", "EllipticalClusters"]}),
            ("pca", {"time_complexity": "O(n*d\u00b2)", "patterns": ["LinearReduction", "Unsupervised"], "best_for": ["Visualization", "NoiseReduction", "FeatureExtraction"]}),
        ]
        for name, meta in ml_algorithms:
            kg.add_algorithm(name, meta)

        extra_algorithms = [
            ("floyd_warshall", {"time_complexity": "O(V^3)", "patterns": ["GraphTraversal", "DynamicProgramming", "AllPairs"], "best_for": ["AllPairsShortest", "DenseGraph", "TransitiveClosure", "WeightedGraph"]}),
            ("genetic_algorithm", {"time_complexity": "O(gens*pop*fitness)", "patterns": ["Evolutionary", "PopulationBased", "Crossover", "Mutation"], "best_for": ["Combinatorial", "MultiModal", "LargeSearchSpace"]}),
            ("hill_climbing", {"time_complexity": "O(iterations)", "patterns": ["LocalSearch", "Iterative", "GradientFree"], "best_for": ["ContinuousOptimization", "LocalOptimum", "FastConvergence"]}),
            ("pso", {"time_complexity": "O(iterations*particles)", "patterns": ["SwarmIntelligence", "PopulationBased", "Continuous"], "best_for": ["ContinuousOptimization", "MultiObjective", "GlobalOptimum"]}),
            ("aco", {"time_complexity": "O(iterations*ants*nodes)", "patterns": ["SwarmIntelligence", "PheromoneBased", "Combinatorial"], "best_for": ["TSP", "VehicleRouting", "NetworkRouting"]}),
        ]
        for name, meta in extra_algorithms:
            kg.add_algorithm(name, meta)

        kg.add_problem_type("sorting", [
            "quicksort", "insertion_sort", "merge_sort",
            "timsort", "radix_sort", "heap_sort",
        ])
        kg.add_problem_type("pathfinding", [
            "dijkstra", "a_star", "bfs_path", "floyd_warshall",
        ])
        kg.add_problem_type("optimization", [
            "greedy_knapsack", "simulated_annealing",
            "genetic_algorithm", "hill_climbing", "pso", "aco",
        ])
        kg.add_problem_type("search", [
            "quicksort", "a_star", "bfs_path",
        ])
        kg.add_problem_type("regression", [
            "linear_regression", "ridge", "lasso",
        ])
        kg.add_problem_type("classification", [
            "logistic_regression", "knn", "svm", "gaussian_nb",
            "random_forest_classification", "xgboost_classification", "lightgbm_classification",
        ])
        kg.add_problem_type("clustering", [
            "kmeans", "dbscan", "gmm",
        ])

        nlp_algorithms = [
            ("word2vec_trainer", {"time_complexity": "O(V*E)", "patterns": ["EmbeddingTraining", "NeuralNetwork", "Unsupervised"], "best_for": ["CustomEmbeddings", "DomainSpecific", "SmallCorpus"]}),
            ("frequency_arithmetic", {"time_complexity": "O(N)", "patterns": ["FrequencyBased", "SimpleArithmetic", "Deterministic"], "best_for": ["WordAnalogy", "Lab1", "SimpleNLP"]}),
            ("word_vector_arithmetic", {"time_complexity": "O(V)", "patterns": ["EmbeddingBased", "Analogy", "Pretrained"], "best_for": ["WordAnalogy", "SemanticRelationships", "GloVe"]}),
            ("embedding_visualization", {"time_complexity": "O(V*D^2)", "patterns": ["Visualization", "PCA", "TSNE", "DimensionalityReduction"], "best_for": ["EmbeddingVisualization", "ClusterAnalysis", "Lab2"]}),
            ("sentiment_analysis", {"time_complexity": "O(L)", "patterns": ["Classification", "Transformers", "Sentiment"], "best_for": ["SentimentAnalysis", "OpinionMining", "ReviewAnalysis", "Lab6"]}),
            ("text_summarization", {"time_complexity": "O(L^2)", "patterns": ["Seq2Seq", "Transformers", "Summarization"], "best_for": ["TextSummarization", "DocumentSummary", "Lab7"]}),
            ("rag_retrieval", {"time_complexity": "O(N*D)", "patterns": ["Retrieval", "Embeddings", "SimilaritySearch"], "best_for": ["DocumentRetrieval", "QuestionAnswering", "RAG", "Lab10"]}),
            ("semantic_search", {"time_complexity": "O(N*D)", "patterns": ["Retrieval", "Embeddings", "Search"], "best_for": ["SemanticSearch", "DocumentMatching", "DuplicateDetection"]}),
            ("prompt_enrichment", {"time_complexity": "O(V)", "patterns": ["PromptEngineering", "Embeddings", "Enrichment"], "best_for": ["PromptEnhancement", "QueryExpansion", "Lab4"]}),
            ("creative_generation", {"time_complexity": "O(V)", "patterns": ["Generation", "Embeddings", "Creative"], "best_for": ["CreativeWriting", "StoryGeneration", "Lab5"]}),
            ("word_expansion", {"time_complexity": "O(D*V)", "patterns": ["Expansion", "Embeddings", "RelatedTerms"], "best_for": ["KeywordExpansion", "QueryBroadening", "Brainstorming"]}),
        ]
        for name, meta in nlp_algorithms:
            kg.add_algorithm(name, meta)

        kg.add_problem_type("nlp", [
            "word2vec_trainer",
            "frequency_arithmetic",
            "word_vector_arithmetic",
            "embedding_visualization",
            "sentiment_analysis",
            "text_summarization",
            "rag_retrieval",
            "semantic_search",
            "prompt_enrichment",
            "creative_generation",
            "word_expansion",
        ])

        kg.add_problem_type("ml", [
            "linear_regression", "ridge", "lasso",
            "logistic_regression", "knn", "svm", "gaussian_nb",
            "random_forest_classification", "xgboost_classification", "lightgbm_classification",
            "kmeans", "dbscan", "gmm",
            "pca",
            "word2vec_trainer",
            "frequency_arithmetic",
            "word_vector_arithmetic",
            "embedding_visualization",
        ])

        kg.add_problem_type("generation", [
            "creative_generation",
            "prompt_enrichment",
        ])

        kg.add_problem_type("image_processing", [
            "gaussian_blur", "median_filter", "bilateral_filter",
            "sobel_edge", "clahe",
        ])

        kg.add_problem_type("computer_vision", [
            "sobel_edge", "gaussian_blur", "median_filter",
            "bilateral_filter", "clahe",
        ])

        self._add_cross_domain_edges(kg)

    def _add_cross_domain_edges(self, kg: Any) -> None:
        pattern_to_problems = {
            "DivideAndConquer":   ["optimization", "search"],
            "Greedy":             ["optimization"],
            "HeuristicSearch":    ["pathfinding", "search"],
            "ComparisonSort":     ["sorting", "search"],
        }
        for pattern, problems in pattern_to_problems.items():
            for pt in problems:
                if kg.graph.has_node(pattern) and kg.graph.has_node(pt):
                    kg.graph.add_edge(pattern, pt,
                                      relation="APPLICABLE_TO", weight=0.8)

        similar_pairs = [
            ("sorting", "search"),
            ("pathfinding", "search"),
            ("optimization", "search"),
        ]
        for p1, p2 in similar_pairs:
            if kg.graph.has_node(p1) and kg.graph.has_node(p2):
                kg.graph.add_edge(p1, p2,
                                  relation="SIMILAR_TO", weight=0.6)
                kg.graph.add_edge(p2, p1,
                                  relation="SIMILAR_TO", weight=0.6)

