from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)

_POSITIVE = {"good", "great", "excellent", "love", "like", "amazing", "happy", "best", "wonderful"}
_NEGATIVE = {"bad", "terrible", "awful", "hate", "worst", "sad", "poor", "angry", "horrible"}


@algorithm(AlgorithmSpec(
    name="sentiment_analyzer",
    task=ProblemTask.SENTIMENT_ANALYSIS,
    domain=Domain.NLP,
    complexity=Complexity("O(n)", "O(1)", "n", "1"),
    principles=frozenset({"ml", "lexical_classification"}),
    deterministic=True, exact=False,
    tags=frozenset({"nlp", "sentiment", "optional"}),
))
class SentimentAnalyzer(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        text = spec.inputs.get("text", "")
        if not text:
            return {"label": "NEUTRAL", "score": 0.0}
        try:
            from transformers import pipeline
            result = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")(text)[0]
            return {"label": result["label"], "score": result["score"]}
        except Exception:
            words = {w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()}
            pos = len(words & _POSITIVE)
            neg = len(words & _NEGATIVE)
            if pos > neg:
                return {"label": "POSITIVE", "score": pos / max(pos + neg, 1)}
            if neg > pos:
                return {"label": "NEGATIVE", "score": neg / max(pos + neg, 1)}
            return {"label": "NEUTRAL", "score": 0.0}


@algorithm(AlgorithmSpec(
    name="text_summarizer",
    task=ProblemTask.TEXT_SUMMARIZATION,
    domain=Domain.NLP,
    complexity=Complexity("O(n)", "O(n)", "n", "n"),
    principles=frozenset({"compression", "text_analysis"}),
    deterministic=True, exact=False,
    tags=frozenset({"nlp", "summarization", "optional"}),
))
class TextSummarizer(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        text = spec.inputs.get("text", "")
        max_sentences = int(spec.inputs.get("max_sentences", 2))
        if not text or len(text.split()) < 20:
            return text
        try:
            from transformers import pipeline
            result = pipeline("summarization", model="facebook/bart-large-cnn")(text, max_length=150, min_length=30)[0]
            return result["summary_text"]
        except Exception:
            import re
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            return " ".join(sentences[:max_sentences]) if sentences else " ".join(text.split()[:50])
