"""
algorithms/nlp/generation.py

Text generation algorithms:
- Prompt enrichment (Lab 4)
- Creative sentence generation (Lab 5)
- Word expansion
"""

import logging
import random
from typing import Any

from aalgoi.algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class PromptEnricher(Algorithm):
    """
    Enrich a prompt with similar words from embeddings.

    Implements Lab 4.

    Input:
        {
            "prompt": "Explain the role of a defendant",
            "seed_word": "defendant",
            "top_n": 5,
            "style": "formal"
        }

    Output:
        {
            "original_prompt": "Explain the role of a defendant",
            "enriched_prompt": "Explain the role of a defendant. Include terms like argued, verdict, suing, judge, case to provide more detail.",
            "similar_words": ["argued", "verdict", "suing", "judge", "case"],
            "valid": true
        }
    """

    name = "prompt_enrichment"

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        return isinstance(output_data, dict) and output_data.get("valid", False)

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "prompt", "enrichment", "generation", "lab4"]
        self.best_for = ["PromptEnhancement", "QueryExpansion", "ContentEnrichment"]
        self.time_complexity = "O(V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["EmbeddingBased", "PromptEngineering", "SemanticSimilarity"]
        self.problem_types = ["NLP", "GENERATION"]

    def process(self, data: Any) -> dict:
        prompt = data.get("prompt", "")
        seed_word = data.get("seed_word", "")
        top_n = data.get("top_n", 5)
        style = data.get("style", "formal")

        if not prompt:
            return {"valid": False, "error": "Prompt required"}

        similar_words = self._get_similar_words(seed_word or self._extract_keyword(prompt), top_n)

        if not similar_words:
            return {
                "original_prompt": prompt,
                "enriched_prompt": prompt,
                "similar_words": [],
                "valid": True,
                "enriched": False
            }

        enriched = self._enrich_prompt(prompt, similar_words, style)

        return {
            "original_prompt": prompt,
            "enriched_prompt": enriched,
            "similar_words": similar_words,
            "seed_word": seed_word,
            "style": style,
            "valid": True,
            "enriched": True,
            "algorithm": self.name
        }

    def _extract_keyword(self, prompt: str) -> str:
        words = prompt.lower().split()
        words = [w.strip(".,!?;:") for w in words if len(w) > 3]
        return words[0] if words else ""

    def _get_similar_words(self, word: str, top_n: int) -> list[str]:
        if not word:
            return []

        try:
            import gensim.downloader as api

            if not self._loaded:
                self.model = api.load("glove-wiki-gigaword-100")
                self._loaded = True

            if word.lower() in self.model:
                similar = self.model.most_similar(word.lower(), topn=top_n)
                return [w[0] for w in similar]

        except Exception:
            pass

        domain_words = {
            'defendant': ['argued', 'verdict', 'suing', 'judge', 'case', 'plaintiff', 'attorney'],
            'court': ['judge', 'lawyer', 'trial', 'evidence', 'jury', 'hearing', 'ruling'],
            'justice': ['fairness', 'law', 'rights', 'punishment', 'order', 'equity'],
            'contract': ['agreement', 'binding', 'legal', 'terms', 'obligation', 'party'],
            'crime': ['offense', 'illegal', 'penalty', 'violation', 'criminal', 'law'],
        }

        for key, words in domain_words.items():
            if key in word.lower():
                return words[:top_n]

        return []

    def _enrich_prompt(self, prompt: str, similar_words: list[str], style: str) -> str:
        if style == "formal":
            return "{} Include terms like {} to provide more detail.".format(prompt, ', '.join(similar_words[:3]))
        elif style == "casual":
            return "{} Think about stuff like {}.".format(prompt, ', '.join(similar_words[:3]))
        elif style == "technical":
            return "{} Consider technical aspects: {}.".format(prompt, ', '.join(similar_words[:5]))
        else:
            return "{} Related concepts: {}.".format(prompt, ', '.join(similar_words[:3]))


class CreativeSentenceGenerator(Algorithm):
    """
    Generate creative sentences using word embeddings.

    Implements Lab 5.

    Input:
        {
            "seed_word": "ocean",
            "num_sentences": 4,
            "style": "story"
        }

    Output:
        {
            "sentences": [...],
            "paragraph": "Combined paragraph...",
            "seed_word": "ocean",
            "similar_words": ["sea", "water", "coast", ...],
            "valid": true
        }
    """

    name = "creative_generation"

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "generation", "creative", "embeddings", "lab5"]
        self.best_for = ["CreativeWriting", "StoryGeneration", "ContentCreation"]
        self.time_complexity = "O(V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["EmbeddingBased", "CreativeGeneration", "RandomWalk"]
        self.problem_types = ["NLP", "GENERATION"]

    def process(self, data: Any) -> dict:
        seed_word = data.get("seed_word", "")
        num_sentences = data.get("num_sentences", 4)
        style = data.get("style", "story")

        if not seed_word:
            return {"valid": False, "error": "Seed word required"}

        similar_words = self._get_similar_words(seed_word, 5)

        if not similar_words:
            return {"valid": False, "error": f"'{seed_word}' not found in vocabulary"}

        sentences = self._generate_sentences(seed_word, similar_words, num_sentences, style)

        return {
            "sentences": sentences,
            "paragraph": " ".join(sentences),
            "seed_word": seed_word,
            "similar_words": similar_words,
            "style": style,
            "valid": True,
            "algorithm": self.name
        }

    def _get_similar_words(self, word: str, top_n: int) -> list[str]:
        try:
            import gensim.downloader as api

            if not self._loaded:
                self.model = api.load("glove-wiki-gigaword-100")
                self._loaded = True

            if word.lower() in self.model:
                similar = self.model.most_similar(word.lower(), topn=top_n)
                return [w[0] for w in similar]

        except Exception:
            pass

        fallback_words = {
            'ocean': ['sea', 'water', 'coast', 'marine', 'wave'],
            'mountain': ['peak', 'climb', 'summit', 'ridge', 'valley'],
            'forest': ['tree', 'woods', 'wildlife', 'nature', 'green'],
        }
        return fallback_words.get(word.lower(), [])

    def _generate_sentences(self, seed: str, similar: list[str], n: int, style: str) -> list[str]:
        templates = {
            'story': [
                f"The {seed} was surrounded by {similar[0]} and {similar[1]}.",
                f"People often associate {seed} with {similar[2]} and {similar[3]}.",
                f"In the land of {seed}, {similar[4]} was a common sight.",
                f"A story about {seed} would be incomplete without {similar[1]} and {similar[3]}.",
                f"The {similar[0]} whispered tales of the {seed}.",
                f"Beyond the {seed}, there lay {similar[2]} unknown to many.",
            ],
            'poem': [
                f"{seed.capitalize()} dances with {similar[0]},",
                f"Where {similar[1]} meets the sky.",
                f"{similar[2].capitalize()} sings of {seed},",
                f"Under watchful {similar[3]}.",
            ],
            'description': [
                f"The {seed} is characterized by its {similar[0]}.",
                f"Notable features include {similar[1]} and {similar[2]}.",
                f"Many consider {similar[3]} essential to understanding {seed}.",
                f"The relationship between {seed} and {similar[4]} is significant.",
            ]
        }

        chosen_templates = templates.get(style, templates['story'])

        sentences = []
        for _ in range(n):
            sentences.append(random.choice(chosen_templates))

        return sentences


class WordExpander(Algorithm):
    """
    Expand a word into related terms and concepts.

    Input:
        {
            "word": "machine learning",
            "depth": 2,
            "top_n": 5
        }

    Output:
        {
            "original": "machine learning",
            "expanded": {
                "level_1": ["AI", "neural network", "deep learning", ...],
                "level_2": ["artificial intelligence", "CNN", "RNN", ...]
            },
            "all_terms": [...],
            "valid": true
        }
    """

    name = "word_expansion"

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "expansion", "synonyms", "related"]
        self.best_for = ["KeywordExpansion", "QueryBroadening", "Brainstorming"]
        self.time_complexity = "O(D * V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["GraphTraversal", "SynonymExpansion"]
        self.problem_types = ["NLP", "GENERATION"]

    def process(self, data: Any) -> dict:
        word = data.get("word", "")
        depth = data.get("depth", 2)
        top_n = data.get("top_n", 5)

        if not word:
            return {"valid": False, "error": "Word required"}

        if not self._loaded:
            self._load_model()

        expanded = {}
        all_terms = set()
        current_words = [word.lower()]

        for level in range(1, depth + 1):
            level_terms = []
            for w in current_words:
                similar = self._get_similar(w, top_n)
                level_terms.extend(similar)

            level_terms = list(set(level_terms))
            expanded["level_%d" % level] = level_terms
            all_terms.update(level_terms)
            current_words = level_terms

        return {
            "original": word,
            "expanded": expanded,
            "all_terms": list(all_terms),
            "depth": depth,
            "valid": True,
            "algorithm": self.name
        }

    def _load_model(self) -> bool:
        try:
            import gensim.downloader as api

            self.model = api.load("glove-wiki-gigaword-100")
            self._loaded = True
            return True
        except Exception:
            return False

    def _get_similar(self, word: str, top_n: int) -> list[str]:
        if not self.model:
            return []

        try:
            if word in self.model:
                similar = self.model.most_similar(word, topn=top_n)
                return [w[0] for w in similar]
        except Exception:
            pass

        return []


