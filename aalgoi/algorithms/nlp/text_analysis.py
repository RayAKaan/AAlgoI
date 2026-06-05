"""
algorithms/nlp/text_analysis.py

Text analysis algorithms:
- Sentiment analysis (Lab 6)
- Text summarization (Lab 7)
"""

import logging
from typing import Any

from aalgoi.algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class SentimentAnalyzer(Algorithm):
    """
    Analyze sentiment of text using transformers.

    Implements Lab 6.

    Input:
        {
            "texts": ["I love this!", "This is terrible."],
            "model": "distilbert-base-uncased-finetuned-sst-2-english"
        }

    Output:
        {
            "results": [
                {"text": "I love this!", "label": "POSITIVE", "score": 0.9998},
                {"text": "This is terrible.", "label": "NEGATIVE", "score": 0.9997}
            ],
            "summary": {"positive": 1, "negative": 1, "neutral": 0},
            "valid": true
        }
    """

    name = "sentiment_analysis"

    def __init__(self):
        self.pipeline = None
        self._loaded = False
        self.tags = ["nlp", "sentiment", "classification", "transformers", "lab6"]
        self.best_for = ["SentimentAnalysis", "OpinionMining", "ReviewAnalysis"]
        self.time_complexity = "O(L)"
        self.space_complexity = "O(M)"
        self.patterns = ["TransformerBased", "SequenceClassification"]
        self.problem_types = ["NLP", "CLASSIFICATION"]

    def process(self, data: Any) -> dict:
        texts = data.get("texts", [])
        model = data.get("model", "distilbert-base-uncased-finetuned-sst-2-english")

        if not texts:
            return {"valid": False, "error": "No texts provided"}

        if isinstance(texts, str):
            texts = [texts]

        if not self._loaded:
            success = self._load_pipeline(model)
            if not success:
                return {"valid": False, "error": "Could not load sentiment model"}

        try:
            results_raw = self.pipeline(texts)

            results = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for i, r in enumerate(results_raw):
                label = r["label"]
                score = r["score"]

                results.append({
                    "text": texts[i],
                    "label": label,
                    "score": float(score)
                })

                if label == "POSITIVE":
                    positive_count += 1
                elif label == "NEGATIVE":
                    negative_count += 1
                else:
                    neutral_count += 1

            return {
                "results": results,
                "summary": {
                    "positive": positive_count,
                    "negative": negative_count,
                    "neutral": neutral_count,
                    "total": len(texts)
                },
                "model": model,
                "valid": True,
                "algorithm": self.name
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _load_pipeline(self, model: str) -> bool:
        try:
            from transformers import pipeline

            self.pipeline = pipeline("sentiment-analysis", model=model)
            self._loaded = True
            return True

        except ImportError:
            logger.error("transformers not installed")
            return False
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)
            return False


class TextSummarizer(Algorithm):
    """
    Summarize long text using transformers.

    Implements Lab 7.

    Input:
        {
            "text": "Long text to summarize...",
            "max_length": 100,
            "min_length": 30,
            "model": "sshleifer/distilbart-cnn-12-6"
        }

    Output:
        {
            "summary": "Summarized text...",
            "original_length": 500,
            "summary_length": 80,
            "compression_ratio": 0.16,
            "valid": true
        }
    """

    name = "text_summarization"

    def __init__(self):
        self.pipeline = None
        self._loaded = False
        self.tags = ["nlp", "summarization", "transformers", "lab7"]
        self.best_for = ["TextSummarization", "DocumentSummary", "ArticleCondensation"]
        self.time_complexity = "O(L^2)"
        self.space_complexity = "O(L)"
        self.patterns = ["TransformerBased", "Seq2Seq", "AbstractiveSummarization"]
        self.problem_types = ["NLP", "SUMMARIZATION"]

    def process(self, data: Any) -> dict:
        text = data.get("text", "")
        max_length = data.get("max_length", 100)
        min_length = data.get("min_length", 30)
        model = data.get("model", "sshleifer/distilbart-cnn-12-6")

        if not text:
            return {"valid": False, "error": "No text provided"}

        if not self._loaded:
            success = self._load_pipeline(model)
            if not success:
                return {"valid": False, "error": "Could not load summarization model"}

        try:
            result = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summary = result[0]["summary_text"]
            original_length = len(text)
            summary_length = len(summary)

            return {
                "summary": summary,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": summary_length / original_length if original_length > 0 else 0,
                "model": model,
                "valid": True,
                "algorithm": self.name
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _load_pipeline(self, model: str) -> bool:
        try:
            from transformers import pipeline

            self.pipeline = pipeline("summarization", model=model)
            self._loaded = True
            return True

        except ImportError:
            logger.error("transformers not installed")
            return False
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)
            return False


