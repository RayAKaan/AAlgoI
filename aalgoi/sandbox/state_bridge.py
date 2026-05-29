"""
aalgoi.sandbox.state_bridge — Encode (query, data) into fixed-size state vectors.

Vocabulary built from actual registry metadata (algorithm names, tags,
problem_types) — so "sort" correctly activates the sorting cluster.
ContextEngine.analyze() for data features — real statistical profiling.

State vector layout:
  [query_vec (VOCAB_SIZE) | data_vec (DATA_DIM) | algo_mask (N)]
"""

from __future__ import annotations
import re
import numpy as np
from typing import Dict, Any

VOCAB_SIZE = 128
DATA_DIM   = 32


class StateBridge:
    """Bridges (query, data) to fixed numpy state vector."""

    def __init__(self, registry: Dict):
        self.registry   = registry
        self.algo_names = sorted(registry.keys())
        self.num_algos  = len(self.algo_names)
        self.state_size = VOCAB_SIZE + DATA_DIM + self.num_algos

        self._vocab: Dict[str, int] = {}
        self._build_vocabulary()

        self._context_engine = None
        try:
            from core.context_engine import ContextEngine
            self._context_engine = ContextEngine()
        except Exception:
            pass

    # ── Vocabulary ─────────────────────────────────────────────────────────

    def _build_vocabulary(self):
        terms = set()

        for name, algo in self.registry.items():
            for tok in re.split(r"[_\-\s]+", name.lower()):
                if tok:
                    terms.add(tok)

            try:
                meta = algo.describe() if hasattr(algo, "describe") else {}
                for field in ["tags", "best_for", "problem_types", "patterns"]:
                    for item in meta.get(field, []):
                        for tok in re.split(r"[_\-\s]+", str(item).lower()):
                            if tok:
                                terms.add(tok)
            except Exception:
                pass

        common = [
            "sort", "search", "find", "path", "route", "cluster", "classify",
            "predict", "optimize", "minimize", "maximize", "reduce", "embed",
            "compress", "detect", "extract", "summarize", "generate", "analyze",
            "shortest", "longest", "fastest", "nearest", "similar", "group",
            "rank", "filter", "transform", "encode", "decode", "train", "fit",
        ]
        terms.update(common)

        for i, term in enumerate(sorted(terms)):
            if i >= VOCAB_SIZE:
                break
            self._vocab[term] = i

    # ── Public API ─────────────────────────────────────────────────────────

    def encode(self, query: str, data: Any) -> np.ndarray:
        q = self._encode_query(query)
        d = self._encode_data(data)
        m = np.ones(self.num_algos, dtype=np.float32)
        return np.concatenate([q, d, m]).astype(np.float32)

    def algo_idx(self, name: str) -> int:
        return self.algo_names.index(name)

    def idx_to_algo(self, idx: int) -> str:
        return self.algo_names[idx]

    # ── Query encoding ─────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Vocabulary-based term-frequency encoding with partial matching.

        For each word in the query:
          - Exact match in vocab -> weight 1.0
          - Partial match (word in vocab_term or vocab_term in word) -> weight 0.6
          - Character bigram match -> weight 0.3
        """
        vec   = np.zeros(VOCAB_SIZE, dtype=np.float32)
        words = re.split(r"\W+", query.lower())
        words = [w for w in words if w]

        for word in words:
            if word in self._vocab:
                vec[self._vocab[word]] += 1.0
                continue

            for term, idx in self._vocab.items():
                if word in term or term in word:
                    vec[idx] += 0.6
                elif len(word) >= 3 and len(term) >= 3:
                    w_bi = {word[i:i+2] for i in range(len(word) - 1)}
                    t_bi = {term[i:i+2] for i in range(len(term) - 1)}
                    overlap = len(w_bi & t_bi) / max(len(w_bi | t_bi), 1)
                    if overlap > 0.4:
                        vec[idx] += 0.3 * overlap

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── Data feature encoding ───────────────────────────────────────────────

    def _encode_data(self, data: Any) -> np.ndarray:
        if self._context_engine is not None:
            try:
                return self._encode_via_context_engine(data)
            except Exception:
                pass
        return self._encode_manual(data)

    def _encode_via_context_engine(self, data: Any) -> np.ndarray:
        ctx = self._context_engine.analyze(data, task_type="unknown")
        features = ctx.get("features", {})
        vec = np.zeros(DATA_DIM, dtype=np.float32)

        ordered_keys = [
            "data_size_log", "is_numeric", "is_sorted", "is_nearly_sorted",
            "cpu_free", "memory_free", "has_repeated", "inversion_ratio",
            "is_classification", "is_regression", "is_clustering",
            "n_samples_log", "n_features_log", "n_classes_log",
            "mean_normalized", "std_normalized", "skewness", "range_normalized",
        ]

        for i, key in enumerate(ordered_keys):
            if i >= DATA_DIM:
                break
            val = features.get(key, 0.0)
            vec[i] = float(np.clip(val, -3.0, 3.0)) / 3.0

        offset = len(ordered_keys)
        struct = self._encode_manual(data)
        remaining = DATA_DIM - offset
        if remaining > 0:
            vec[offset:offset + remaining] = struct[:remaining]

        return vec

    def _encode_manual(self, data: Any) -> np.ndarray:
        vec = np.zeros(DATA_DIM, dtype=np.float32)
        if data is None:
            return vec

        if not isinstance(data, dict):
            data = {"data": data}

        for v in data.values():
            if isinstance(v, (list, np.ndarray)):
                arr = list(v)
                vec[0] = 1.0
                vec[1] = float(min(len(arr) / 1000.0, 1.0))
                if arr and isinstance(arr[0], (list, np.ndarray)):
                    vec[2] = 1.0
                    w = len(arr[0]) if hasattr(arr[0], "__len__") else 0
                    vec[3] = float(min(w / 50.0, 1.0))
                    if w > 1:
                        vec[4] = 1.0
                elif arr and isinstance(arr[0], (int, float)):
                    sample = arr[:min(len(arr), 100)]
                    vec[5] = 1.0 if sample == sorted(sample) else 0.0
                    vec[6] = float(np.std(sample) / max(abs(np.mean(sample)) + 1e-6, 1))
                break

        vec[7]  = 1.0 if "graph" in data else 0.0
        vec[8]  = 1.0 if "X_train" in data else 0.0
        vec[9]  = 1.0 if "y_train" in data else 0.0
        vec[10] = 1.0 if "n_clusters" in data else 0.0
        vec[11] = 1.0 if "n_components" in data else 0.0
        vec[12] = 1.0 if ("text" in data or "texts" in data) else 0.0
        vec[13] = 1.0 if "corpus" in data else 0.0
        vec[14] = 1.0 if "image" in data else 0.0
        vec[15] = 1.0 if "items" in data else 0.0
        vec[16] = 1.0 if "capacity" in data else 0.0
        vec[17] = 1.0 if ("query" in data and "document" in data) else 0.0
        vec[18] = 1.0 if "operation" in data else 0.0
        vec[19] = 1.0 if "target" in data else 0.0
        vec[20] = 1.0 if "start" in data else 0.0
        vec[21] = 1.0 if "end" in data else 0.0
        vec[22] = float(len(data.keys()) / 20.0)

        if "y_train" in data:
            y = data["y_train"]
            if isinstance(y, (list, np.ndarray)) and len(y) > 0:
                first = y[0]
                if isinstance(first, str):
                    vec[23] = 1.0
                elif isinstance(first, (int, float)):
                    unique_ratio = len(set(y)) / len(y)
                    vec[24] = 1.0 if unique_ratio < 0.1 else 0.0
                    vec[25] = 1.0 if unique_ratio > 0.5 else 0.0

        return vec
