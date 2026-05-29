"""
algorithms/nlp/generation.py

Text generation algorithms:
- Prompt enrichment (Lab 4)
- Creative sentence generation (Lab 5)
- Word expansion
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging
import random

from algorithms.base import Algorithm

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

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and output_data.get("valid", False)

    def __init__(self):
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "prompt", "enrichment", "generation", "lab4"]
        self.best_for = ["PromptEnhancement", "QueryExpansion", "ContentEnrichment"]
        self.time_complexity = "O(V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["EmbeddingBased", "PromptEngineering", "SemanticSimilarity"]
        self.problem_types = ["NLP", "GENERATION"]

    def process(self, data: Any) -> Dict:
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

    def _get_similar_words(self, word: str, top_n: int) -> List[str]:
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

    def _enrich_prompt(self, prompt: str, similar_words: List[str], style: str) -> str:
        if style == "formal":
            return "%s Include terms like %s to provide more detail." % (prompt, ', '.join(similar_words[:3]))
        elif style == "casual":
            return "%s Think about stuff like %s." % (prompt, ', '.join(similar_words[:3]))
        elif style == "technical":
            return "%s Consider technical aspects: %s." % (prompt, ', '.join(similar_words[:5]))
        else:
            return "%s Related concepts: %s." % (prompt, ', '.join(similar_words[:3]))


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

    def __init__(self):
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "generation", "creative", "embeddings", "lab5"]
        self.best_for = ["CreativeWriting", "StoryGeneration", "ContentCreation"]
        self.time_complexity = "O(V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["EmbeddingBased", "CreativeGeneration", "RandomWalk"]
        self.problem_types = ["NLP", "GENERATION"]

    def process(self, data: Any) -> Dict:
        seed_word = data.get("seed_word", "")
        num_sentences = data.get("num_sentences", 4)
        style = data.get("style", "story")

        if not seed_word:
            return {"valid": False, "error": "Seed word required"}

        similar_words = self._get_similar_words(seed_word, 5)

        if not similar_words:
            return {"valid": False, "error": "'%s' not found in vocabulary" % seed_word}

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

    def _get_similar_words(self, word: str, top_n: int) -> List[str]:
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

    def _generate_sentences(self, seed: str, similar: List[str], n: int, style: str) -> List[str]:
        templates = {
            'story': [
                "The %s was surrounded by %s and %s." % (seed, similar[0], similar[1]),
                "People often associate %s with %s and %s." % (seed, similar[2], similar[3]),
                "In the land of %s, %s was a common sight." % (seed, similar[4]),
                "A story about %s would be incomplete without %s and %s." % (seed, similar[1], similar[3]),
                "The %s whispered tales of the %s." % (similar[0], seed),
                "Beyond the %s, there lay %s unknown to many." % (seed, similar[2]),
            ],
            'poem': [
                "%s dances with %s," % (seed.capitalize(), similar[0]),
                "Where %s meets the sky." % similar[1],
                "%s sings of %s," % (similar[2].capitalize(), seed),
                "Under watchful %s." % similar[3],
            ],
            'description': [
                "The %s is characterized by its %s." % (seed, similar[0]),
                "Notable features include %s and %s." % (similar[1], similar[2]),
                "Many consider %s essential to understanding %s." % (similar[3], seed),
                "The relationship between %s and %s is significant." % (seed, similar[4]),
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

    def __init__(self):
        self.model = None
        self._loaded = False
        self.tags = ["nlp", "expansion", "synonyms", "related"]
        self.best_for = ["KeywordExpansion", "QueryBroadening", "Brainstorming"]
        self.time_complexity = "O(D * V)"
        self.space_complexity = "O(V * D)"
        self.patterns = ["GraphTraversal", "SynonymExpansion"]
        self.problem_types = ["NLP", "GENERATION"]

    def process(self, data: Any) -> Dict:
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

    def _get_similar(self, word: str, top_n: int) -> List[str]:
        if not self.model:
            return []

        try:
            if word in self.model:
                similar = self.model.most_similar(word, topn=top_n)
                return [w[0] for w in similar]
        except Exception:
            pass

        return []


