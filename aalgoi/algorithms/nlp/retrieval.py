"""
algorithms/nlp/retrieval.py

Retrieval algorithms:
- RAG (Retrieval-Augmented Generation) - Lab 10
- Semantic search
"""

import logging
from typing import Any

from aalgoi.algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class RAGRetriever(Algorithm):
    """
    Retrieve relevant passages from documents using semantic search.

    Implements Lab 10's retrieval component.

    Input:
        {
            "document": "Full text of document...",
            "query": "What is the punishment for theft?",
            "top_k": 3,
            "chunk_size": 500,
            "model": "paraphrase-MiniLM-L6-v2"
        }

    Output:
        {
            "passages": ["Relevant passage 1", "Relevant passage 2", ...],
            "scores": [0.85, 0.72, ...],
            "query": "What is the punishment for theft?",
            "total_passages": 150,
            "valid": true
        }
    """

    name = "rag_retrieval"

    def __init__(self):
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "rag", "retrieval", "semantic_search", "lab10"]
        self.best_for = ["DocumentRetrieval", "QuestionAnswering", "InformationRetrieval"]
        self.time_complexity = "O(N * D)"
        self.space_complexity = "O(N * D)"
        self.patterns = ["EmbeddingBased", "RetrievalAugmented", "ChunkAndEmbed"]
        self.problem_types = ["NLP", "RETRIEVAL"]

    def process(self, data: Any) -> dict:
        document = data.get("document", "")
        query = data.get("query", "")
        top_k = data.get("top_k", 3)
        chunk_size = data.get("chunk_size", 500)
        model_name = data.get("model", "paraphrase-MiniLM-L6-v2")

        if not document or not query:
            return {"valid": False, "error": "Document and query required"}

        if not self._loaded:
            success = self._load_model(model_name)
            if not success:
                return {"valid": False, "error": "Could not load embedding model"}

        try:
            passages = self._split_into_passages(document, chunk_size)

            if not passages:
                return {"valid": False, "error": "Could not split document into passages"}

            passage_embeddings = self.model.encode(passages)
            query_embedding = self.model.encode([query])

            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, passage_embeddings)[0]

            top_indices = similarities.argsort()[::-1][:top_k]

            results = []
            scores = []
            for idx in top_indices:
                results.append(passages[idx])
                scores.append(float(similarities[idx]))

            return {
                "passages": results,
                "scores": scores,
                "query": query,
                "total_passages": len(passages),
                "model": model_name,
                "valid": True,
                "algorithm": self.name
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _load_model(self, model_name: str) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._loaded = True
            return True

        except ImportError:
            logger.error("sentence-transformers not installed")
            return False
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def _split_into_passages(self, text: str, max_length: int = 500) -> list[str]:
        text = text.replace('\n\n', ' ').replace('\n', ' ')
        sentences = text.split('. ')

        passages = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) < max_length:
                current += sentence + ". "
            else:
                if current:
                    passages.append(current.strip())
                current = sentence + ". "

        if current:
            passages.append(current.strip())

        return passages


class SemanticSearcher(Algorithm):
    """
    Search for semantically similar texts in a corpus.

    Input:
        {
            "corpus": ["doc1", "doc2", ...],
            "query": "search query",
            "top_k": 5
        }

    Output:
        {
            "results": [
                {"text": "doc1", "score": 0.95, "index": 0},
                ...
            ],
            "valid": true
        }
    """

    name = "semantic_search"

    def __init__(self):
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "search", "semantic", "similarity"]
        self.best_for = ["SemanticSearch", "DocumentMatching", "DuplicateDetection"]
        self.time_complexity = "O(N * D)"
        self.space_complexity = "O(N * D)"
        self.patterns = ["EmbeddingBased", "SimilaritySearch", "DenseRetrieval"]
        self.problem_types = ["NLP", "RETRIEVAL"]

    def process(self, data: Any) -> dict:
        corpus = data.get("corpus", [])
        query = data.get("query", "")
        top_k = data.get("top_k", 5)

        if not corpus or not query:
            return {"valid": False, "error": "Corpus and query required"}

        if not self._loaded:
            success = self._load_model()
            if not success:
                return {"valid": False, "error": "Could not load model"}

        try:
            corpus_embeddings = self.model.encode(corpus)
            query_embedding = self.model.encode([query])

            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

            top_indices = similarities.argsort()[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "text": corpus[idx],
                    "score": float(similarities[idx]),
                    "index": int(idx)
                })

            return {
                "results": results,
                "query": query,
                "corpus_size": len(corpus),
                "valid": True,
                "algorithm": self.name
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _load_model(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self._loaded = True
            return True

        except ImportError:
            return False


