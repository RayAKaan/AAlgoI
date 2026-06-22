from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="sentiment_analyzer",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.NLP,
    complexity=Complexity("O(n)", "O(1)", "n", "1"),
    principles=frozenset({"ml"}),
    deterministic=False, exact=False,
    tags=frozenset({"nlp", "optional"}),
))
class SentimentAnalyzer(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            from transformers import pipeline
        except ImportError:
            raise OptionalDependencyMissing("nlp", "transformers")
        text = spec.inputs.get("text", "")
        if not text:
            return {"label": "NEUTRAL", "score": 0.0}
        result = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")(text)[0]
        return {"label": result["label"], "score": result["score"]}


@algorithm(AlgorithmSpec(
    name="text_summarizer",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.NLP,
    complexity=Complexity("O(n)", "O(1)", "n", "1"),
    principles=frozenset({"ml"}),
    deterministic=False, exact=False,
    tags=frozenset({"nlp", "optional"}),
))
class TextSummarizer(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            from transformers import pipeline
        except ImportError:
            raise OptionalDependencyMissing("nlp", "transformers")
        text = spec.inputs.get("text", "")
        if not text or len(text.split()) < 20:
            return text
        result = pipeline("summarization", model="facebook/bart-large-cnn")(text, max_length=150, min_length=30)[0]
        return result["summary_text"]
