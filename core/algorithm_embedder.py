import numpy as np
import torch
from typing import Dict, Optional, List

COMPLEXITY_MAP = {
    "O(1)":               0.0,
    "O(log n)":           0.1,
    "O(n)":               0.3,
    "O(n log n)":         0.5,
    "O(n^2)":             0.7,
    "O(n^3)":             0.85,
    "O(2^n)":             1.0,
    "O(nk)":              0.45,
    "O(V log V)":         0.5,
    "O(V+E)":             0.4,
    "O(E)":               0.35,
    "O(V)":               0.3,
    "O(iterations)":      0.7,
    "O(n * d^2)":         0.7,
    "O(n * window * dim)":0.5,
    "O(n + k)":           0.35,
    "Unknown":            0.5,
}

PROBLEM_TYPE_IDX = {
    "SORTING":           0,
    "PATHFINDING":       1,
    "OPTIMIZATION":      2,
    "CLUSTERING":        3,
    "CLASSIFICATION":    4,
    "REGRESSION":        5,
    "EMBEDDING":         6,
    "IMAGE_PROCESSING":  7,
    "GRAPH":             8,
    "SEARCH":            9,
    "COMPRESSION":       10,
    "UNKNOWN":           11,
}


class AlgorithmEmbedder:
    EMBED_DIM = 32

    def __init__(self):
        self._cache: Dict[str, torch.Tensor] = {}

    def get_embedding(self, algo_name: str) -> Optional[torch.Tensor]:
        return self._cache.get(algo_name)

    def add_embedding(self, algo_name: str, embedding: torch.Tensor):
        self._cache[algo_name] = embedding

    def embed_algorithm(self, algorithm) -> torch.Tensor:
        if algorithm.name in self._cache:
            return self._cache[algorithm.name]

        vec = np.zeros(self.EMBED_DIM, dtype=np.float32)

        complexity = getattr(algorithm, 'time_complexity', 'Unknown')
        space_cplx = getattr(algorithm, 'space_complexity', 'Unknown')
        tags = getattr(algorithm, 'tags', [])
        best_for = getattr(algorithm, 'best_for', [])

        vec[0] = COMPLEXITY_MAP.get(complexity, 0.5)
        vec[1] = COMPLEXITY_MAP.get(complexity, 0.5)
        vec[2] = COMPLEXITY_MAP.get(complexity, 0.5)
        vec[3] = COMPLEXITY_MAP.get(space_cplx, 0.3)

        vec[4] = 1.0 if 'stable' in tags else 0.0
        vec[5] = 1.0 if 'in_place' in tags else 0.0

        problem_types = self._infer_problem_types(algorithm)
        for pt in problem_types:
            idx = PROBLEM_TYPE_IDX.get(pt.upper())
            if idx is not None:
                vec[6 + idx] = 1.0

        all_hints = best_for + tags

        vec[18] = 1.0 if any('small' in h for h in all_hints) else 0.0
        vec[19] = 1.0 if any('large' in h for h in all_hints) else 0.0
        vec[20] = 1.0 if any('sorted' in h for h in all_hints) else 0.0
        vec[21] = 1.0 if any('random' in h for h in all_hints) else 0.0
        vec[22] = 1.0 if any('sparse' in h for h in all_hints) else 0.0
        vec[23] = 1.0 if any('dense' in h for h in all_hints) else 0.0

        vec[24] = 1.0 if any(t in tags for t in ['cpu_intensive', 'combinatorial']) else 0.0
        vec[25] = 1.0 if 'memory_constrained' in best_for else 0.0
        vec[26] = 0.0
        vec[27] = 1.0 if 'parallelizable' in tags else 0.0

        vec[28] = 0.0 if any(t in tags for t in ['approximation', 'metaheuristic']) else 1.0
        vec[29] = 1.0
        vec[30] = 0.0
        vec[31] = 0.5

        embed = torch.tensor(vec, dtype=torch.float32)
        self._cache[algorithm.name] = embed
        return embed

    def embed_all(self, registry: dict) -> torch.Tensor:
        embeddings = []
        for name, algo in registry.items():
            embed = self.embed_algorithm(algo)
            embeddings.append(embed)
        if not embeddings:
            return torch.zeros(1, self.EMBED_DIM)
        return torch.stack(embeddings, dim=0)

    def get_all_embeddings(self, registry: dict) -> torch.Tensor:
        embeddings = []
        for name, algo in registry.items():
            if name in self._cache:
                embeddings.append(self._cache[name])
            else:
                embeddings.append(self.embed_algorithm(algo))
        return torch.stack(embeddings, dim=0)

    def _infer_problem_types(self, algorithm) -> List[str]:
        name = algorithm.name.lower()
        tags = [t.lower() for t in getattr(algorithm, 'tags', [])]
        all_text = [name] + tags
        inferred = []

        def kw_in(keywords):
            return any(kw in text for text in all_text for kw in keywords)

        if kw_in(['sort']):
            inferred.append('SORTING')
        if kw_in(['pathfinding', 'dijkstra', 'astar', 'bfs_path', 'breadth_first', 'weighted_graph', 'unweighted_graph', 'shortest_path']):
            inferred.append('PATHFINDING')
        if kw_in(['optimization', 'knapsack', 'anneal', 'greedy', 'metaheuristic', 'combinatorial', 'resource_allocation']):
            inferred.append('OPTIMIZATION')
        if kw_in(['kmeans', 'dbscan', 'cluster']):
            inferred.append('CLUSTERING')
        if kw_in(['forest', 'classify', 'svm', 'random_forest']):
            inferred.append('CLASSIFICATION')
        if kw_in(['regress']):
            inferred.append('REGRESSION')
        if kw_in(['word2vec', 'embedding', 'pca', 'tsne', 'dimensionality_reduction', 'semantic']):
            inferred.append('EMBEDDING')
        if kw_in(['blur', 'filter', 'edge', 'clahe', 'gaussian', 'median', 'sobel']):
            inferred.append('IMAGE_PROCESSING')
        if kw_in(['search', 'lookup', 'rabin_karp', 'binary_search', 'linear_search', 'locate']):
            inferred.append('SEARCH')
        if kw_in(['nlp', 'language', 'corpus', 'sentence', 'word2vec']):
            inferred.append('NLP')
        if kw_in(['graph']):
            inferred.append('GRAPH')
        if kw_in(['compression', 'zip', 'huffman', 'lzw']):
            inferred.append('COMPRESSION')

        if not inferred:
            module_path = type(algorithm).__module__
            if 'image_processing' in module_path:
                inferred.append('IMAGE_PROCESSING')
            elif 'embeddings' in module_path:
                inferred.append('EMBEDDING')
            elif 'ml' in module_path:
                inferred.append('CLASSIFICATION')
            else:
                inferred.append('UNKNOWN')

        return inferred
