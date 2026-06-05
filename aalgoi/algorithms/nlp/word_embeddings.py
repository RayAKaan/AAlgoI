"""
algorithms/nlp/word_embeddings.py

Word embedding algorithms:
- Training Word2Vec from corpus
- Frequency-based vector arithmetic (Lab 1)
- Pre-trained vector arithmetic (GloVe)
- Embedding visualization (Lab 2)
"""

import logging
from collections import Counter
from typing import Any

import numpy as np

from aalgoi.algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class Word2VecTrainer(Algorithm):
    """
    Train Word2Vec embeddings on a corpus.

    Input:
        {
            "corpus": [["word1", "word2", ...], ["sentence2", ...]],
            "vector_size": 100,
            "window": 5,
            "min_count": 1,
            "epochs": 10
        }

    Output:
        {
            "vocab_size": 1500,
            "vector_size": 100,
            "trained": True,
            "model_path": "/path/to/saved/model"
        }
    """

    name = "word2vec_trainer"

    def __init__(self):
        self.model = None
        self.tags = ["nlp", "embeddings", "word2vec", "unsupervised"]
        self.best_for = ["TrainEmbeddings", "CustomVocabulary", "DomainSpecific"]
        self.time_complexity = "O(V * E)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["EmbeddingTraining", "NeuralNetwork", "ContextPrediction"]
        self.problem_types = ["NLP", "EMBEDDINGS"]

    def process(self, data: Any) -> dict:
        corpus = data.get("corpus", [])
        vector_size = data.get("vector_size", 100)
        window = data.get("window", 5)
        min_count = data.get("min_count", 1)
        epochs = data.get("epochs", 10)
        save_path = data.get("save_path", None)

        if not corpus:
            return {"trained": False, "error": "No corpus provided"}

        if isinstance(corpus[0], str):
            corpus = [sentence.lower().split() for sentence in corpus]

        try:
            from gensim.models import Word2Vec

            self.model = Word2Vec(
                sentences=corpus,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4,
                epochs=epochs,
                seed=42
            )

            vocab_size = len(self.model.wv)

            if save_path:
                self.model.save(save_path)

            return {
                "trained": True,
                "vocab_size": vocab_size,
                "vector_size": vector_size,
                "window": window,
                "corpus_size": len(corpus),
                "model_path": save_path,
                "algorithm": self.name
            }

        except ImportError:
            return {
                "trained": False,
                "error": "gensim not installed. Run: pip install gensim"
            }
        except Exception as e:
            logger.error("Word2Vec training failed: %s", e)
            return {"trained": False, "error": str(e)}

    def get_vector(self, word: str) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            return self.model.wv[word.lower()]
        except KeyError:
            return None

    def get_similar(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        try:
            return self.model.wv.most_similar(word.lower(), topn=top_n)
        except KeyError:
            return []


class FrequencyVectorArithmetic(Algorithm):
    """
    Perform word arithmetic using frequency-based vectors.

    Implements Lab 1's approach using word frequencies.

    Input:
        {
            "corpus": ["king is a strong man", "queen is a wise woman", ...],
            "operation": "king - man + woman",
            "remove_stopwords": true
        }

    Output:
        {
            "result": "queen",
            "closest_words": ["queen", "princess", "woman", ...],
            "operation": "king - man + woman",
            "valid": true
        }
    """

    name = "frequency_arithmetic"

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and output_data.get("valid", False)

    def __init__(self):
        self.tags = ["nlp", "word_arithmetic", "frequency", "deterministic"]
        self.best_for = ["WordAnalogy", "FrequencyAnalysis", "Lab1", "FrequencyArithmetic"]
        self.time_complexity = "O(N)"
        self.space_complexity = "O(V)"
        self.patterns = ["FrequencyBased", "SimpleArithmetic", "Analogy", "Frequency"]
        self.problem_types = ["NLP", "WORD_ARITHMETIC"]

    def process(self, data: Any) -> dict:
        corpus = data.get("corpus", [])
        operation = data.get("operation", "")
        remove_stopwords = data.get("remove_stopwords", True)
        top_k = data.get("top_k", 5)

        if not corpus or not operation:
            return {"valid": False, "error": "Corpus and operation required"}

        if isinstance(corpus[0], str):
            tokenized = [sentence.lower().split() for sentence in corpus]
        else:
            tokenized = corpus

        if remove_stopwords:
            tokenized = self._remove_stopwords(tokenized)

        all_words = [word for sentence in tokenized for word in sentence]
        freq = Counter(all_words)

        positive, negative = self._parse_operation(operation)

        all_input = positive + negative
        missing = [w for w in all_input if w not in freq]
        if missing:
            return {
                "valid": False,
                "error": f"Words not in vocabulary: {missing}",
                "vocabulary_size": len(freq)
            }

        result_freq = 0
        for word in positive:
            result_freq += freq[word]
        for word in negative:
            result_freq -= freq[word]

        distances = {}
        for word, count in freq.items():
            if word not in all_input:
                distances[word] = abs(count - result_freq)

        sorted_words = sorted(distances.items(), key=lambda x: x[1])
        closest = [w[0] for w in sorted_words[:top_k]]

        similarities = {}
        if positive:
            word1 = positive[0]
            for word in closest[:top_k]:
                sim = self._cosine_similarity_freq(freq, word1, word)
                similarities[word] = sim

        return {
            "result": closest[0] if closest else None,
            "result_frequency": result_freq,
            "closest_words": closest,
            "similarities": similarities,
            "vocabulary": dict(freq.most_common(20)),
            "vocabulary_size": len(freq),
            "operation": operation,
            "positive_words": positive,
            "negative_words": negative,
            "valid": True,
            "algorithm": self.name
        }

    def _remove_stopwords(self, tokenized: list[list[str]]) -> list[list[str]]:
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            return [
                [word for word in sentence if word.lower() not in stop_words]
                for sentence in tokenized
            ]
        except Exception:
            stop_words = {'is', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            return [
                [word for word in sentence if word.lower() not in stop_words]
                for sentence in tokenized
            ]

    def _parse_operation(self, expr: str) -> tuple[list[str], list[str]]:
        tokens = expr.lower().split()
        positive = []
        negative = []
        sign = "+"

        for token in tokens:
            if token == "+":
                sign = "+"
            elif token == "-":
                sign = "-"
            elif token not in ["plus", "minus"]:
                word = token.strip(".,!?;:")
                if sign == "+":
                    positive.append(word)
                else:
                    negative.append(word)

        return positive, negative

    def _cosine_similarity_freq(self, freq: Counter, word1: str, word2: str) -> float:
        v1 = freq.get(word1, 0)
        v2 = freq.get(word2, 0)

        if v1 == 0 or v2 == 0:
            return 0.0

        return float(v1 * v2) / (float(np.sqrt(v1) * np.sqrt(v2)))


class WordVectorArithmetic(Algorithm):
    """
    Perform word arithmetic using pre-trained embeddings (GloVe/Word2Vec).

    Uses semantic embeddings for more accurate analogies.

    Input:
        {
            "operation": "king - man + woman",
            "embedding_model": "glove-wiki-gigaword-100",
            "top_k": 5
        }

    Output:
        {
            "result": ["queen", "princess", "monarch", ...],
            "scores": [0.85, 0.72, ...],
            "operation": "king - man + woman",
            "valid": true
        }
    """

    name = "word_vector_arithmetic"

    def __init__(self):
        self.model = None
        self.model_name = None
        self._loaded = False
        self.tags = ["nlp", "word_arithmetic", "embeddings", "glove", "analogy"]
        self.best_for = ["WordAnalogy", "SemanticRelationships", "EmbeddingExploration"]
        self.time_complexity = "O(V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["EmbeddingBased", "Analogy", "Pretrained"]
        self.problem_types = ["NLP", "WORD_ARITHMETIC"]

    def process(self, data: Any) -> dict:
        operation = data.get("operation", "")
        model_name = data.get("embedding_model", "glove-wiki-gigaword-100")
        top_k = data.get("top_k", 5)

        if not operation:
            return {"valid": False, "error": "Operation required"}

        if not self._loaded or self.model_name != model_name:
            success = self._load_model(model_name)
            if not success:
                return {"valid": False, "error": "Could not load model: " + model_name}

        positive, negative = self._parse_operation(operation)

        all_input = positive + negative
        missing = [w for w in all_input if w not in self.model]
        if missing:
            return {
                "valid": False,
                "error": f"Words not in vocabulary: {missing}",
                "model": self.model_name
            }

        try:
            result = self.model.most_similar(
                positive=positive,
                negative=negative,
                topn=top_k
            )

            words = [r[0] for r in result]
            scores = [float(r[1]) for r in result]

            return {
                "result": words,
                "scores": scores,
                "operation": operation,
                "positive_words": positive,
                "negative_words": negative,
                "model": self.model_name,
                "valid": True,
                "algorithm": self.name
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _load_model(self, model_name: str) -> bool:
        try:
            import gensim.downloader as api

            try:
                available = api.info()['models']
            except Exception:
                available = {}

            if model_name in available:
                self.model = api.load(model_name)
                self.model_name = model_name
                self._loaded = True
                return True

            if model_name.endswith('.txt') or model_name.endswith('.vec'):
                from gensim.models import KeyedVectors
                self.model = KeyedVectors.load_word2vec_format(model_name, binary=False)
                self.model_name = model_name
                self._loaded = True
                return True

            return False

        except Exception as e:
            logger.error("Failed to load model %s: %s", model_name, e)
            return False

    def _parse_operation(self, expr: str) -> tuple[list[str], list[str]]:
        tokens = expr.lower().split()
        positive = []
        negative = []
        sign = "+"

        for token in tokens:
            if token == "+":
                sign = "+"
            elif token == "-":
                sign = "-"
            else:
                word = token.strip(".,!?;:")
                if sign == "+":
                    positive.append(word)
                else:
                    negative.append(word)

        return positive, negative


class EmbeddingVisualization(Algorithm):
    """
    Visualize word embeddings in 2D/3D using PCA or t-SNE.

    Implements Lab 2 and Lab 3's visualization component.

    Input:
        {
            "words": ["basketball", "soccer", "football", ...],
            "corpus": ["basketball is a sport", ...],
            "method": "pca",
            "dimensions": 2,
            "pretrained": "glove-wiki-gigaword-100"
        }

    Output:
        {
            "coordinates": [[x1, y1], [x2, y2], ...],
            "words": ["word1", "word2", ...],
            "method": "pca",
            "dimensions": 2,
            "explained_variance": 0.85,
            "plot_data": {...}
        }
    """

    name = "embedding_visualization"

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and output_data.get("valid", False)

    def __init__(self):
        self.embeddings = None
        self.word_list = []
        self.tags = ["nlp", "visualization", "pca", "tsne", "embeddings", "lab2"]
        self.best_for = ["Embedding_Visualization", "ClusterAnalysis", "SemanticMapping"]
        self.time_complexity = "O(V * D^2)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["DimensionalityReduction", "Visualization"]
        self.problem_types = ["NLP", "VISUALIZATION"]

    def process(self, data: Any) -> dict:
        words = data.get("words", [])
        corpus = data.get("corpus", None)
        method = data.get("method", "pca")
        dimensions = data.get("dimensions", 2)
        pretrained = data.get("pretrained", "glove-wiki-gigaword-100")

        if not words and not corpus:
            return {"valid": False, "error": "Words or corpus required"}

        if corpus:
            embeddings = self._train_embeddings(corpus, words)
        elif pretrained:
            embeddings = self._load_pretrained(words, pretrained)
        else:
            return {"valid": False, "error": "No embedding source specified"}

        if embeddings is None or len(embeddings) == 0:
            return {"valid": False, "error": "Could not create embeddings"}

        if method == "pca":
            reduced, variance = self._pca_reduce(embeddings, dimensions)
        elif method == "tsne":
            reduced, variance = self._tsne_reduce(embeddings, dimensions)
        else:
            return {"valid": False, "error": "Unknown method: " + method}

        coordinates = reduced.tolist()
        distances = self._compute_distances(coordinates, words)
        clusters = self._find_clusters(coordinates, words, threshold=0.3)

        return {
            "coordinates": coordinates,
            "words": words,
            "method": method,
            "dimensions": dimensions,
            "explained_variance": variance,
            "distances": distances,
            "clusters": clusters,
            "valid": True,
            "algorithm": self.name
        }

    def _train_embeddings(self, corpus: list[str], words: list[str]) -> np.ndarray | None:
        try:
            from gensim.models import Word2Vec

            if isinstance(corpus[0], str):
                tokenized = [sentence.lower().split() for sentence in corpus]
            else:
                tokenized = corpus

            model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1, workers=1, epochs=10)

            embeddings = []
            for word in words:
                if word.lower() in model.wv:
                    embeddings.append(model.wv[word.lower()])
                else:
                    embeddings.append(np.random.randn(100))

            return np.array(embeddings)

        except Exception as e:
            logger.error("Failed to train embeddings: %s", e)
            return None

    def _load_pretrained(self, words: list[str], model_name: str) -> np.ndarray | None:
        try:
            import gensim.downloader as api

            model = api.load(model_name)

            embeddings = []
            for word in words:
                if word.lower() in model:
                    embeddings.append(model[word.lower()])
                else:
                    embeddings.append(np.random.randn(100))

            return np.array(embeddings)

        except Exception as e:
            logger.error("Failed to load pretrained embeddings: %s", e)
            return np.random.randn(len(words), 100)

    def _pca_reduce(self, embeddings: np.ndarray, dimensions: int) -> tuple[np.ndarray, float]:
        from sklearn.decomposition import PCA

        n_components = min(dimensions, embeddings.shape[0], embeddings.shape[1])

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)

        variance = float(sum(pca.explained_variance_ratio_))

        if reduced.shape[1] < dimensions:
            padding = np.zeros((reduced.shape[0], dimensions - reduced.shape[1]))
            reduced = np.hstack([reduced, padding])

        return reduced, variance

    def _tsne_reduce(self, embeddings: np.ndarray, dimensions: int) -> tuple[np.ndarray, None]:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=dimensions, random_state=42, perplexity=min(30, len(embeddings) - 1))
        reduced = tsne.fit_transform(embeddings)

        return reduced, None

    def _compute_distances(self, coordinates: list[list[float]], words: list[str]) -> dict[str, float]:
        distances = {}
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i < j:
                    d = sum((coordinates[i][k] - coordinates[j][k]) ** 2 for k in range(len(coordinates[i])))
                    dist = float(np.sqrt(d))
                    distances[f"{w1}-{w2}"] = dist

        return distances

    def _find_clusters(self, coordinates: list[list[float]], words: list[str], threshold: float) -> list[list[str]]:
        n = len(words)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            cluster = [words[i]]
            visited[i] = True

            for j in range(i + 1, n):
                if not visited[j]:
                    d = sum((coordinates[i][k] - coordinates[j][k]) ** 2 for k in range(len(coordinates[i])))
                    dist = float(np.sqrt(d))

                    max_d = 1.0
                    for k in range(n):
                        dd = sum((coordinates[i][t] - coordinates[k][t]) ** 2 for t in range(len(coordinates[i])))
                        max_d = max(max_d, float(np.sqrt(dd)))

                    norm_dist = dist / max_d

                    if norm_dist < threshold:
                        cluster.append(words[j])
                        visited[j] = True

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters


