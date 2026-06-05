
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class VectorKnowledgeBase:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.collection_name = self.config.get("collection_name", "aalgoi_memory")
        self.max_size = self.config.get("max_size", 100000)
        self.records: List[Dict] = []
        self._index: Dict[str, List[int]] = defaultdict(list)
        self._chroma_client = None
        self._chroma_collection = None
        self._use_chromadb = False
        self._init_chromadb()

    def _init_chromadb(self):
        try:
            import chromadb
            self._chroma_client = chromadb.Client()
            try:
                self._chroma_collection = self._chroma_client.get_collection(self.collection_name)
            except ValueError:
                self._chroma_collection = self._chroma_client.create_collection(self.collection_name)
            self._use_chromadb = True
        except (ImportError, Exception):
            pass

    def store(self, context: Dict, algorithm_names: List[str], metrics: Dict):
        record = {
            "timestamp": time.time(),
            "context": {
                "features": context.get("features", {}),
                "data_profile": {
                    "type": context.get("data_profile", {}).get("type"),
                    "size": context.get("data_profile", {}).get("size"),
                    "patterns": context.get("data_profile", {}).get("patterns", {})
                },
                "constraints": context.get("constraints", {})
            },
            "algorithms": algorithm_names,
            "metrics": {
                "wall_time_ms": metrics.get("wall_time_ms"),
                "quality_score": metrics.get("quality_score"),
                "success": metrics.get("success"),
                "within_budget": metrics.get("within_budget"),
                "score": metrics.get("score", 0)
            }
        }

        self.records.append(record)
        idx = len(self.records) - 1
        for algo in algorithm_names:
            self._index[algo].append(idx)

        if self._use_chromadb and self._chroma_collection is not None:
            try:
                doc_id = f"rec_{idx}"
                doc_text = json.dumps({
                    "type": context.get("data_profile", {}).get("type"),
                    "size": context.get("data_profile", {}).get("size"),
                    "is_sorted": context.get("data_profile", {}).get("patterns", {}).get("is_sorted"),
                    "is_nearly_sorted": context.get("data_profile", {}).get("patterns", {}).get("is_nearly_sorted"),
                    "priority": context.get("constraints", {}).get("priority"),
                    "algorithms": algorithm_names
                })
                metadata = {
                    "algorithms": ",".join(algorithm_names),
                    "score": str(metrics.get("score", 0)),
                    "success": str(metrics.get("success", True))
                }
                self._chroma_collection.add(
                    documents=[doc_text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            except Exception:
                pass

        if len(self.records) > self.max_size:
            self._prune_old_records()

    def query_similar(self, context: Dict, top_k: int = 5) -> List[Dict]:
        if self._use_chromadb and self._chroma_collection is not None:
            try:
                query_text = json.dumps({
                    "type": context.get("data_profile", {}).get("type"),
                    "size": context.get("data_profile", {}).get("size"),
                    "is_sorted": context.get("data_profile", {}).get("patterns", {}).get("is_sorted"),
                    "is_nearly_sorted": context.get("data_profile", {}).get("patterns", {}).get("is_nearly_sorted"),
                    "priority": context.get("constraints", {}).get("priority")
                })
                results = self._chroma_collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, len(self.records))
                )
                if results and results.get("metadatas") and results["metadatas"][0]:
                    return self._chroma_to_records(results)
            except Exception:
                pass

        return self._fallback_similarity_search(context, top_k)

    def _chroma_to_records(self, results: Dict) -> List[Dict]:
        records = []
        for i, meta in enumerate(results["metadatas"][0]):
            rec = {
                "algorithms": meta.get("algorithms", "").split(",") if meta.get("algorithms") else [],
                "metrics": {
                    "score": float(meta.get("score", 0)),
                    "success": meta.get("success", "True") == "True"
                }
            }
            if results.get("documents") and results["documents"][0] and i < len(results["documents"][0]):
                try:
                    doc = json.loads(results["documents"][0][i])
                    rec["context"] = {"data_profile": doc}
                except (json.JSONDecodeError, IndexError):
                    pass
            records.append(rec)
        return records

    def _fallback_similarity_search(self, context: Dict, top_k: int) -> List[Dict]:
        if not self.records:
            return []

        current_features = context.get("features", {})
        current_vec = self._features_to_vector(current_features)

        similarities = []
        for record in self.records:
            record_features = record.get("context", {}).get("features", {})
            record_vec = self._features_to_vector(record_features)
            similarity = self._cosine_similarity(current_vec, record_vec)
            similarities.append((similarity, record))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in similarities[:top_k]]

    def penalize(self, algorithm_name: str, context: Dict):
        penalty_record = {
            "timestamp": time.time(),
            "context": {"features": context.get("features", {})},
            "algorithms": [algorithm_name],
            "metrics": {
                "score": -1.0,
                "success": False,
                "penalty": True
            }
        }
        self.records.append(penalty_record)
        self._index[algorithm_name].append(len(self.records) - 1)

    def discount_old_records(self, factor: float = 0.5):
        cutoff = time.time() - 86400 * 7
        for record in self.records:
            if record.get("timestamp", 0) < cutoff:
                metrics = record.get("metrics", {})
                score = metrics.get("score", 0)
                if score > 0:
                    metrics["score"] = score * factor

    def get_algorithm_stats(self, algorithm_name: str) -> Dict[str, Any]:
        indices = self._index.get(algorithm_name, [])
        if not indices:
            return {}

        records = [self.records[i] for i in indices]
        times = [r["metrics"].get("wall_time_ms", 0) for r in records if r["metrics"].get("wall_time_ms") is not None]
        qualities = [r["metrics"].get("quality_score", 0) for r in records if r["metrics"].get("quality_score") is not None]
        scores = [r["metrics"].get("score", 0) for r in records if r["metrics"].get("score") is not None]

        return {
            "total_executions": len(records),
            "avg_time_ms": np.mean(times) if times else 0,
            "avg_quality": np.mean(qualities) if qualities else 0,
            "avg_score": np.mean(scores) if scores else 0,
            "success_rate": sum(1 for r in records if r["metrics"].get("success")) / len(records) if records else 0,
            "last_used": max(r.get("timestamp", 0) for r in records) if records else None
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        return {
            algo: self.get_algorithm_stats(algo)
            for algo in self._index.keys()
        }

    def get_best_algorithm(self, context: Dict, metric: str = "quality_score") -> Optional[str]:
        similar = self.query_similar(context, top_k=20)
        if not similar:
            return None

        unique_algos = set()
        for record in similar:
            for algo in record.get("algorithms", []):
                if algo:
                    unique_algos.add(algo)

        if len(unique_algos) == 1:
            return list(unique_algos)[0]

        algo_scores = defaultdict(list)
        for record in similar:
            for algo in record.get("algorithms", []):
                score = record.get("metrics", {}).get(metric)
                if score is not None:
                    algo_scores[algo].append(score)

        if not algo_scores:
            return None

        best_algo = max(algo_scores.keys(), key=lambda a: np.mean(algo_scores[a]))
        return best_algo

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        feature_names = [
            "data_size_log", "is_numeric", "is_sorted", "is_nearly_sorted",
            "cpu_free", "mem_free_ratio", "cpu_count", "time_budget_norm",
            "priority_speed", "priority_accuracy"
        ]
        return np.array([features.get(f, 0.0) for f in feature_names])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _prune_old_records(self):
        keep_count = self.max_size // 2
        self.records = self.records[-keep_count:]
        self._index = defaultdict(list)
        for i, record in enumerate(self.records):
            for algo in record.get("algorithms", []):
                self._index[algo].append(i)


    def merge(self, global_knowledge: Dict):
        """
        Merge knowledge from the federated network.
        Integrate remote algorithm performance metrics.
        """
        if not global_knowledge:
            return

        import time

        for algo_name, metrics in global_knowledge.items():
            entry = {
                "algorithm": algo_name,
                "score": metrics.get("avg_reward", 0),
                "success_rate": metrics.get("success_rate", 0),
                "context_patterns": metrics.get("context_patterns", []),
                "source": "federated",
                "timestamp": time.time(),
            }
            self.records.append(entry)
            idx = len(self.records) - 1
            self._index[algo_name].append(idx)

        logger.info("Merged %d federated knowledge entries", len(global_knowledge))

    def get_top_performing(self, n: int = 10) -> Dict:
        """Extract top N best performing algorithms for sharing."""
        sorted_records = sorted(
            self.records,
            key=lambda x: x.get("metrics", {}).get("score", x.get("score", 0)),
            reverse=True,
        )[:n]

        return {
            r.get("algorithm", "unknown"): {
                "avg_reward": r.get("metrics", {}).get("score", r.get("score", 0)),
                "success_rate": 1.0
                if r.get("metrics", {}).get("score", 0) > 5
                else 0.5,
                "context_patterns": [],
            }
            for r in sorted_records
            if r.get("algorithm")
        }


KnowledgeBase = VectorKnowledgeBase
