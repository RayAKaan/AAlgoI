
from typing import Any

import numpy as np

from aalgoi.algorithms.base import Algorithm


class Word2VecTrainer(Algorithm):
    """Train custom Word2Vec embeddings on domain-specific corpus."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "word2vec_trainer"
        self.time_complexity = "O(n * window * dim)"
        self.tags = ["nlp", "embeddings", "unsupervised"]
        self.best_for = ["domain_specific_corpus", "semantic_similarity"]
        self.patterns = ["EmbeddingTraining", "NeuralNetwork"]
        self.problem_types = ["NLP", "EMBEDDINGS"]

    def process(self, data: dict) -> dict:
        from gensim.models import Word2Vec

        corpus = data.get("corpus", [])
        vector_size = data.get("vector_size", 100)
        window = data.get("window", 5)
        min_count = data.get("min_count", 1)
        epochs = data.get("epochs", 10)
        domain = data.get("domain", "general")

        sentences = [sentence.split() for sentence in corpus if sentence]

        model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=4,
        )

        analysis = self._analyze_domain_semantics(model, domain)

        return {
            "model": model,
            "vocabulary_size": len(model.wv),
            "vector_size": vector_size,
            "domain": domain,
            "domain_analysis": analysis,
        }

    def _analyze_domain_semantics(self, model: Any, domain: str) -> dict:
        domain_pairs = {
            "legal": [
                ("contract", "agreement"),
                ("plaintiff", "defendant"),
                ("tort", "crime"),
            ],
            "medical": [
                ("diagnosis", "prognosis"),
                ("symptom", "sign"),
                ("treatment", "therapy"),
            ],
            "finance": [
                ("asset", "liability"),
                ("bull", "bear"),
                ("dividend", "yield"),
            ],
            "general": [("king", "queen"), ("man", "woman"), ("good", "bad")],
        }

        pairs = domain_pairs.get(domain, domain_pairs["general"])
        similarities = []

        for w1, w2 in pairs:
            if w1 in model.wv and w2 in model.wv:
                sim = float(model.wv.similarity(w1, w2))
                similarities.append({"word1": w1, "word2": w2, "similarity": sim})

        return {
            "domain": domain,
            "pair_similarities": similarities,
            "avg_similarity": (
                float(np.mean([s["similarity"] for s in similarities]))
                if similarities
                else 0
            ),
            "captures_semantics": len(similarities) > 0,
        }


class PCAReduction(Algorithm):
    """Reduce embedding dimensions via PCA for visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "pca_reduction"
        self.time_complexity = "O(n * d^2)"
        self.tags = ["dimensionality_reduction", "visualization"]
        self.best_for = ["embedding_visualization", "feature_extraction"]
        self.patterns = ["LinearReduction", "Unsupervised"]
        self.problem_types = ["DIMENSIONALITY_REDUCTION"]

    def process(self, data: dict) -> dict:
        from sklearn.decomposition import PCA

        embeddings = np.array(data.get("embeddings", []))
        n_components = data.get("n_components", 2)
        words = data.get("words", [])

        if len(embeddings) == 0:
            return {"error": "No embeddings provided"}

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)

        return {
            "reduced_embeddings": reduced,
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "visualization_data": {
                "coordinates": reduced.tolist(),
                "words": words,
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
            },
            "n_components": n_components,
        }


class TSNEVisualization(Algorithm):
    """t-SNE for 2D visualization of high-dimensional embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "tsne_visualization"
        self.time_complexity = "O(n^2)"
        self.tags = ["visualization", "dimensionality_reduction"]
        self.best_for = ["cluster_visualization", "exploratory_analysis"]
        self.patterns = ["NonLinearReduction", "Probabilistic"]
        self.problem_types = ["VISUALIZATION"]

    def process(self, data: dict) -> dict:
        from sklearn.manifold import TSNE

        embeddings = np.array(data.get("embeddings", []))
        words = data.get("words", [])
        perplexity = data.get("perplexity", 30)

        if len(embeddings) == 0:
            return {"error": "No embeddings provided"}

        effective_perplexity = min(perplexity, len(embeddings) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=42,
        )
        embedded = tsne.fit_transform(embeddings)

        return {
            "tsne_coordinates": embedded.tolist(),
            "visualization_data": {
                "coordinates": embedded.tolist(),
                "words": words,
            },
            "perplexity": effective_perplexity,
            "kl_divergence": float(tsne.kl_divergence_),
        }


class SemanticSimilarityGenerator(Algorithm):
    """Generate semantically similar words using embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "semantic_similarity"
        self.time_complexity = "O(n log n)"
        self.tags = ["nlp", "generation", "semantic_search"]
        self.best_for = ["word_generation", "query_expansion"]
        self.patterns = ["EmbeddingBased", "Similarity"]
        self.problem_types = ["NLP", "SIMILARITY"]

    def process(self, data: dict) -> dict:
        model = data.get("model")
        input_word = data.get("input_word", "")
        topn = data.get("topn", 5)

        if model is None:
            return {"similar_words": [], "error": "No model provided"}

        if input_word not in model.wv:
            return {
                "similar_words": [],
                "error": f"'{input_word}' not in vocabulary",
            }

        similar = model.wv.most_similar(input_word, topn=topn)

        return {
            "input_word": input_word,
            "similar_words": [
                {"word": w, "similarity": float(s)} for w, s in similar
            ],
        }
