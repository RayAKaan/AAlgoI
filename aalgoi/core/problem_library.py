
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from aalgoi.core.problem_spec import ProblemSpec


class ProblemLibrary:
    def __init__(self, collection_name: str = "aalgoi_problems"):
        self.collection_name = collection_name
        self.problems: Dict[str, Dict] = {}
        self._embedding_dim = 128
        self._chroma_collection = None
        self._chroma_client = None
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

            existing = self._chroma_collection.get()
            if existing and existing.get("ids"):
                for i, pid in enumerate(existing["ids"]):
                    meta = existing["metadatas"][i] if existing.get("metadatas") else {}
                    doc = existing["documents"][i] if existing.get("documents") else ""
                    self.problems[pid] = {
                        "metadata": meta,
                        "document": doc,
                        "signature": meta.get("signature", ""),
                        "best_algorithms": meta.get("best_algorithms", "").split(",") if meta.get("best_algorithms") else [],
                        "avg_score": float(meta.get("avg_score", 0)),
                        "total_runs": int(meta.get("total_runs", 0))
                    }
        except (ImportError, Exception):
            pass

    def find_similar(self, problem_spec: ProblemSpec, top_k: int = 5,
                     min_similarity: float = 0.0) -> List[Dict]:
        if not self.problems:
            return []

        query_vec = problem_spec.to_vector()

        if self._use_chromadb and self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_embeddings=[query_vec.tolist()],
                    n_results=min(top_k, len(self.problems))
                )
                if results and results.get("metadatas") and results["metadatas"][0]:
                    return self._chroma_to_results(results)
            except Exception:
                pass

        return self._fallback_search(query_vec, top_k, min_similarity)

    def store_solution(self, problem_spec: ProblemSpec, algorithm_names: List[str],
                       performance: Dict):
        signature = problem_spec.get_signature()
        problem_id = f"{signature}_{int(time.time())}"

        entry = {
            "signature": signature,
            "best_algorithms": algorithm_names,
            "avg_score": performance.get("score", 0),
            "total_runs": 1,
            "last_used": time.time(),
            "timestamp": time.time()
        }

        if problem_id in self.problems:
            existing = self.problems[problem_id]
            entry["total_runs"] = existing.get("total_runs", 0) + 1
            entry["avg_score"] = (existing.get("avg_score", 0) * (entry["total_runs"] - 1) + performance.get("score", 0)) / entry["total_runs"]
            if performance.get("score", 0) > existing.get("avg_score", 0):
                entry["best_algorithms"] = algorithm_names

        self.problems[problem_id] = entry

        if self._use_chromadb and self._chroma_collection is not None:
            try:
                vec = problem_spec.to_vector()
                doc_text = json.dumps(problem_spec.to_dict())
                metadata = {
                    "signature": signature,
                    "best_algorithms": ",".join(algorithm_names),
                    "avg_score": str(performance.get("score", 0)),
                    "total_runs": str(entry["total_runs"]),
                    "problem_type": problem_spec.problem_type.value,
                    "name": problem_spec.name
                }

                if self._has_id(problem_id):
                    self._chroma_collection.update(
                        ids=[problem_id],
                        embeddings=[vec.tolist()],
                        metadatas=[metadata],
                        documents=[doc_text]
                    )
                else:
                    self._chroma_collection.add(
                        ids=[problem_id],
                        embeddings=[vec.tolist()],
                        metadatas=[metadata],
                        documents=[doc_text]
                    )
            except Exception:
                pass

    def _has_id(self, problem_id: str) -> bool:
        try:
            existing = self._chroma_collection.get(ids=[problem_id])
            return bool(existing and existing.get("ids"))
        except Exception:
            return False

    def get_best_algorithms(self, problem_spec: ProblemSpec, top_k: int = 3) -> List[Tuple[str, float]]:
        similar = self.find_similar(problem_spec, top_k=5, min_similarity=0.3)

        algo_scores = defaultdict(list)
        for result in similar:
            algos = result.get("best_algorithms", [])
            score = result.get("avg_score", 0)
            similarity = result.get("similarity", 0.5)
            for algo in algos:
                algo_scores[algo].append(score * similarity)

        if not algo_scores:
            return []

        def weighted_avg(items):
            return sum(items) / len(items)

        ranked = sorted(algo_scores.items(), key=lambda x: weighted_avg(x[1]), reverse=True)
        return [(algo, weighted_avg(scores)) for algo, scores in ranked[:top_k]]

    def get_all_problems(self) -> Dict[str, Dict]:
        return dict(self.problems)

    def get_stats(self) -> Dict[str, Any]:
        if not self.problems:
            return {"total_problems": 0}

        types = defaultdict(int)
        for entry in self.problems.values():
            sig = entry.get("signature", "")
            ptype = sig.split("|")[0] if sig else "unknown"
            types[ptype] += 1

        return {
            "total_problems": len(self.problems),
            "by_type": dict(types),
            "avg_score": np.mean([p.get("avg_score", 0) for p in self.problems.values()]) if self.problems else 0,
            "total_runs": sum(p.get("total_runs", 0) for p in self.problems.values())
        }

    def _chroma_to_results(self, results: Dict) -> List[Dict]:
        output = []
        for i in range(len(results["ids"][0])):
            pid = results["ids"][0][i]
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 0
            similarity = max(0, 1.0 - dist)

            algos_raw = meta.get("best_algorithms", "")
            algos = algos_raw.split(",") if algos_raw else []

            output.append({
                "problem_id": pid,
                "signature": meta.get("signature", ""),
                "best_algorithms": algos,
                "avg_score": float(meta.get("avg_score", 0)),
                "total_runs": int(meta.get("total_runs", 0)),
                "similarity": similarity,
                "problem_type": meta.get("problem_type", ""),
                "name": meta.get("name", "")
            })
        return output

    def _fallback_search(self, query_vec: np.ndarray, top_k: int,
                         min_similarity: float) -> List[Dict]:
        similarities = []
        for pid, entry in self.problems.items():
            stored_vec = self._decode_signature(entry.get("signature", ""))
            if stored_vec is not None:
                sim = self._cosine_similarity(query_vec, stored_vec)
            else:
                sim = 0.5
            similarities.append((sim, pid, entry))

        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, pid, entry in similarities[:top_k]:
            if sim >= min_similarity:
                results.append({
                    "problem_id": pid,
                    "signature": entry.get("signature", ""),
                    "best_algorithms": entry.get("best_algorithms", []),
                    "avg_score": entry.get("avg_score", 0),
                    "total_runs": entry.get("total_runs", 0),
                    "similarity": sim
                })
        return results

    def _decode_signature(self, signature: str) -> Optional[np.ndarray]:
        try:
            rng = np.random.RandomState(hash(signature) & 0xFFFFFFFF)
            return rng.randn(self._embedding_dim)
        except Exception:
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
