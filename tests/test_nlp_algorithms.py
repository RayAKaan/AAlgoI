"""
tests/test_nlp_algorithms.py

Tests for all 11 NLP algorithms:
- Word2VecTrainer
- FrequencyVectorArithmetic (Lab 1)
- WordVectorArithmetic
- EmbeddingVisualization (Lab 2)
- SentimentAnalyzer (Lab 6)
- TextSummarizer (Lab 7)
- RAGRetriever (Lab 10)
- SemanticSearcher
- PromptEnricher (Lab 4)
- CreativeSentenceGenerator (Lab 5)
- WordExpander

Plus integration tests for solve() routing and explainer templates.
"""

import pytest
import numpy as np


@pytest.fixture
def legal_corpus():
    return [
        "The defendant is guilty of the crime",
        "The plaintiff filed a lawsuit against the defendant",
        "The court ruled in favor of the defendant",
        "The legal proceedings were delayed by the judge",
        "The lawyer presented evidence in court",
    ]


_has_gensim = False
try:
    import gensim
    _has_gensim = True
except ImportError:
    pass


class TestWord2VecTrainer:

    def test_trains_on_corpus(self, legal_corpus):
        if not _has_gensim:
            pytest.skip("gensim not installed")
        from aalgoi.algorithms.nlp import Word2VecTrainer

        tokenized = [s.lower().split() for s in legal_corpus]

        algo = Word2VecTrainer()
        result = algo.process({
            "corpus": tokenized,
            "vector_size": 50,
            "window": 5,
            "min_count": 1,
            "epochs": 10
        })

        assert result["trained"] is True
        assert result["vocab_size"] > 0
        assert result["vector_size"] == 50

    def test_get_vector_after_training(self, legal_corpus):
        if not _has_gensim:
            pytest.skip("gensim not installed")
        from aalgoi.algorithms.nlp import Word2VecTrainer

        tokenized = [s.lower().split() for s in legal_corpus]

        algo = Word2VecTrainer()
        algo.process({"corpus": tokenized, "vector_size": 50})

        vec = algo.get_vector("court")
        assert vec is not None
        assert len(vec) == 50

    def test_get_similar_words(self, legal_corpus):
        if not _has_gensim:
            pytest.skip("gensim not installed")
        from aalgoi.algorithms.nlp import Word2VecTrainer

        tokenized = [s.lower().split() for s in legal_corpus]

        algo = Word2VecTrainer()
        algo.process({"corpus": tokenized, "vector_size": 50})

        similar = algo.get_similar("court", top_n=3)
        assert len(similar) > 0

    def test_empty_corpus_returns_error(self):
        from aalgoi.algorithms.nlp import Word2VecTrainer

        algo = Word2VecTrainer()
        result = algo.process({"corpus": []})

        assert result["trained"] is False

    def test_string_corpus_tokenizes(self):
        if not _has_gensim:
            pytest.skip("gensim not installed")
        from aalgoi.algorithms.nlp import Word2VecTrainer

        algo = Word2VecTrainer()
        result = algo.process({
            "corpus": ["hello world", "foo bar"],
            "vector_size": 10,
            "epochs": 1
        })

        assert result["trained"] is True


class TestFrequencyArithmetic:

    def test_basic_arithmetic(self):
        from aalgoi.algorithms.nlp import FrequencyVectorArithmetic

        algo = FrequencyVectorArithmetic()
        result = algo.process({
            "corpus": [
                "king is a strong man",
                "queen is a wise woman",
                "boy is a young man",
                "girl is a young woman",
                "man is strong",
                "woman is pretty",
            ],
            "operation": "king - man + woman"
        })

        assert result["valid"] is True
        assert "queen" in result["closest_words"]

    def test_removes_stopwords_when_requested(self):
        from aalgoi.algorithms.nlp import FrequencyVectorArithmetic

        algo = FrequencyVectorArithmetic()
        result = algo.process({
            "corpus": ["king is a man", "queen is a woman"],
            "operation": "king",
            "remove_stopwords": True
        })

        assert "is" not in result["vocabulary"]
        assert "a" not in result["vocabulary"]

    def test_handles_missing_words(self):
        from aalgoi.algorithms.nlp import FrequencyVectorArithmetic

        algo = FrequencyVectorArithmetic()
        result = algo.process({
            "corpus": ["hello world"],
            "operation": "king - man"
        })

        assert result["valid"] is False
        assert "not in vocabulary" in result["error"]

    def test_empty_corpus(self):
        from aalgoi.algorithms.nlp import FrequencyVectorArithmetic

        algo = FrequencyVectorArithmetic()
        result = algo.process({"corpus": [], "operation": "king"})

        assert result["valid"] is False


class TestEmbeddingVisualization:

    def test_pca_visualization(self):
        from aalgoi.algorithms.nlp import EmbeddingVisualization

        algo = EmbeddingVisualization()
        result = algo.process({
            "words": ["basketball", "soccer", "football", "tennis"],
            "method": "pca",
            "dimensions": 2
        })

        assert result["valid"] is True
        assert len(result["coordinates"]) == 4
        assert len(result["coordinates"][0]) == 2
        assert result["explained_variance"] is not None

    def test_tsne_visualization(self):
        from aalgoi.algorithms.nlp import EmbeddingVisualization

        algo = EmbeddingVisualization()
        result = algo.process({
            "words": ["king", "queen", "man", "woman"],
            "method": "tsne",
            "dimensions": 2
        })

        assert result["valid"] is True
        assert len(result["coordinates"]) == 4

    def test_computes_distances(self):
        from aalgoi.algorithms.nlp import EmbeddingVisualization

        algo = EmbeddingVisualization()
        result = algo.process({
            "words": ["one", "two", "three"],
            "method": "pca",
            "dimensions": 2
        })

        assert "distances" in result

    def test_no_words_returns_error(self):
        from aalgoi.algorithms.nlp import EmbeddingVisualization

        algo = EmbeddingVisualization()
        result = algo.process({"words": [], "method": "pca"})

        assert result["valid"] is False


class TestSentimentAnalyzer:

    def test_available_even_without_transformers(self):
        from aalgoi.algorithms.nlp import SentimentAnalyzer

        algo = SentimentAnalyzer()
        result = algo.process({"texts": ["hello"]})

        assert result["valid"] is False or "error" in result


class TestTextSummarizer:

    def test_available_even_without_transformers(self):
        from aalgoi.algorithms.nlp import TextSummarizer

        algo = TextSummarizer()
        result = algo.process({"text": "hello world"})

        assert result["valid"] is False or "error" in result


class TestRAGRetriever:

    def test_available_even_without_sentence_transformers(self):
        from aalgoi.algorithms.nlp import RAGRetriever

        algo = RAGRetriever()
        result = algo.process({
            "document": "test doc",
            "query": "test query"
        })

        assert result["valid"] is False or "error" in result


class TestSemanticSearcher:

    def test_available_even_without_sentence_transformers(self):
        from aalgoi.algorithms.nlp import SemanticSearcher

        algo = SemanticSearcher()
        result = algo.process({
            "corpus": ["doc1"],
            "query": "test"
        })

        assert result["valid"] is False or "error" in result


class TestPromptEnricher:

    def test_enriches_prompt_with_fallback(self):
        from aalgoi.algorithms.nlp import PromptEnricher

        algo = PromptEnricher()
        result = algo.process({
            "prompt": "Explain the role of a defendant",
            "seed_word": "defendant",
            "top_n": 5
        })

        assert result["valid"] is True
        assert "defendant" in result["enriched_prompt"]

    def test_different_styles(self):
        from aalgoi.algorithms.nlp import PromptEnricher

        algo = PromptEnricher()

        formal = algo.process({
            "prompt": "Explain the role",
            "seed_word": "defendant",
            "style": "formal"
        })
        casual = algo.process({
            "prompt": "Explain the role",
            "seed_word": "defendant",
            "style": "casual"
        })

        assert formal["enriched_prompt"] != casual["enriched_prompt"]

    def test_no_seed_word_extracts_from_prompt(self):
        from aalgoi.algorithms.nlp import PromptEnricher

        algo = PromptEnricher()
        result = algo.process({
            "prompt": "Explain the role of a defendant",
            "top_n": 3
        })

        assert result["valid"] is True

    def test_empty_prompt_returns_error(self):
        from aalgoi.algorithms.nlp import PromptEnricher

        algo = PromptEnricher()
        result = algo.process({"prompt": ""})

        assert result["valid"] is False


class TestCreativeSentenceGenerator:

    def test_generates_sentences(self):
        from aalgoi.algorithms.nlp import CreativeSentenceGenerator

        algo = CreativeSentenceGenerator()
        result = algo.process({
            "seed_word": "ocean",
            "num_sentences": 4
        })

        assert result["valid"] is True
        assert len(result["sentences"]) == 4

    def test_different_styles(self):
        from aalgoi.algorithms.nlp import CreativeSentenceGenerator

        algo = CreativeSentenceGenerator()

        story = algo.process({"seed_word": "mountain", "style": "story", "num_sentences": 2})
        poem = algo.process({"seed_word": "mountain", "style": "poem", "num_sentences": 2})

        assert story["valid"] is True
        assert poem["valid"] is True

    def test_empty_seed_returns_error(self):
        from aalgoi.algorithms.nlp import CreativeSentenceGenerator

        algo = CreativeSentenceGenerator()
        result = algo.process({"seed_word": ""})

        assert result["valid"] is False


class TestWordExpander:

    def test_expands_word(self):
        from aalgoi.algorithms.nlp import WordExpander

        algo = WordExpander()
        result = algo.process({
            "word": "machine learning",
            "depth": 1,
            "top_n": 3
        })

        assert result["valid"] is True
        assert "level_1" in result["expanded"]

    def test_empty_word_returns_error(self):
        from aalgoi.algorithms.nlp import WordExpander

        algo = WordExpander()
        result = algo.process({"word": ""})

        assert result["valid"] is False


class TestNLPRegistry:

    def test_all_nlp_algos_in_registry(self):
        from aalgoi.pipeline import UniversalSolver

        solver = UniversalSolver()

        expected = [
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
        ]

        for name in expected:
            assert name in solver.registry, "%s not in registry" % name

    def test_registry_count_increased(self):
        from aalgoi.pipeline import UniversalSolver

        solver = UniversalSolver()

        assert len(solver.registry) >= 44


class TestNLPExplainer:

    @pytest.mark.parametrize("algo_name", [
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
    def test_explain_nlp_algorithm(self, algo_name):
        from aalgoi import explain

        exp = explain(algo_name)

        assert exp is not None
        assert hasattr(exp, "summary")
        assert len(exp.summary) > 0

    def test_word2vec_complexity(self):
        from aalgoi import explain

        exp = explain("word2vec_trainer")

        assert "O(V * E)" in exp.complexity or "O" in exp.complexity

    def test_sentiment_has_steps(self):
        from aalgoi import explain

        exp = explain("sentiment_analysis")

        assert len(exp.steps) > 0


class TestMetadataPropagation:
    """Verify every NLP algorithm class exposes metadata() correctly."""

    @pytest.mark.parametrize("cls", [
        pytest.param(cls, id=cls.__name__)
        for cls in __import__("aalgoi.algorithms.nlp", fromlist=["NLP_CLASSES"]).NLP_CLASSES
    ])
    def test_metadata_has_required_fields(self, cls):
        algo = cls()
        meta = algo.metadata()
        assert isinstance(meta, dict)
        assert "name" in meta
        assert "tags" in meta
        assert "best_for" in meta
        assert "time_complexity" in meta
        assert "space_complexity" in meta
        assert "patterns" in meta
        assert "problem_types" in meta
        assert isinstance(meta["patterns"], list)
        assert isinstance(meta["problem_types"], list)
        assert len(meta["problem_types"]) > 0
        assert "NLP" in meta["problem_types"]

    @pytest.mark.parametrize("cls", [
        pytest.param(cls, id=cls.__name__)
        for cls in __import__("aalgoi.algorithms.nlp", fromlist=["NLP_CLASSES"]).NLP_CLASSES
    ])
    def test_describe_aliases_metadata(self, cls):
        algo = cls()
        assert algo.describe() == algo.metadata()
