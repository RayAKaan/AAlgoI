"""
algorithms/nlp/__init__.py

NLP algorithms for lab exercises and real-world tasks.

Categories:
- Word Embeddings: training, visualization, arithmetic
- Text Analysis: sentiment, summarization
- Retrieval: RAG, semantic search
- Generation: prompt enrichment, creative writing
"""

from aalgoi.algorithms.nlp.word_embeddings import (
    Word2VecTrainer,
    FrequencyVectorArithmetic,
    WordVectorArithmetic,
    EmbeddingVisualization,
)
from aalgoi.algorithms.nlp.text_analysis import (
    SentimentAnalyzer,
    TextSummarizer,
)
from aalgoi.algorithms.nlp.retrieval import (
    RAGRetriever,
    SemanticSearcher,
)
from aalgoi.algorithms.nlp.generation import (
    PromptEnricher,
    CreativeSentenceGenerator,
    WordExpander,
)

__all__ = [
    "Word2VecTrainer",
    "FrequencyVectorArithmetic",
    "WordVectorArithmetic",
    "EmbeddingVisualization",
    "SentimentAnalyzer",
    "TextSummarizer",
    "RAGRetriever",
    "SemanticSearcher",
    "PromptEnricher",
    "CreativeSentenceGenerator",
    "WordExpander",
]

NLP_CLASSES = [
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
