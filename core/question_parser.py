"""
Lightweight question-to-ProblemSpec converter.
Uses optional DistilBERT + rule patterns for offline inference.
DistilBERT is NOT loaded at import time. It loads on first
_detect_problem_type call that needs it (ambiguous queries only).
"""

import re
import logging
from typing import Dict, Any, Optional, List

from core.problem_spec import ProblemSpec, ProblemType, Objective, Constraint

logger = logging.getLogger(__name__)


class QuestionParser:
    """
    Converts natural language questions into ProblemSpec objects.

    Examples:
        "Sort these numbers fastest possible"
          -> ProblemSpec(type=SORTING, priority=speed)

        "Train Word2Vec on medical corpus with 200 dimensions"
          -> ProblemSpec(type=ML, inputs={corpus, vector_size=200, domain=medical})
    """

    # ── Shorthand aliases (expanded before keyword detection) ──
    SHORTHAND = {
        "sort asc":          "sort ascending",
        "sort desc":         "sort descending",
        "sort fast":         "sort ascending quickly",
        "sort slow":         "sort ascending stably",
        "sort rev":          "sort descending",
        "sort reverse":      "sort descending",
        "sort stable":       "sort ascending stably",
        "sort big":          "sort ascending large data",
        "path":              "find shortest path",
        "shortest":          "find shortest path",
        "route":             "find shortest path",
        "navigate":          "find shortest path",
        "knapsack":          "maximize value within capacity",
        "maximize":          "maximize value",
        "minimize":          "minimize this function",
        "optimize":          "maximize value",
        "schedule":          "schedule tasks optimally",
        "cluster":           "cluster this data",
        "classify":          "classify",
        "regress":           "fit regression model",
        "embed":             "train word embeddings",
        "reduce":            "reduce dimensions",
        "blur":              "blur image",
        "denoise":           "denoise image",
        "edges":             "detect edges in image",
        "enhance":           "enhance this data",
        "why":               "explain algorithm choice",
        "best":              "use best algorithm",
        "fast":              "use fastest algorithm",
        "compare":           "benchmark algorithms",
    }

    ARROW_PATTERN = re.compile(
        r"([A-Za-z0-9_]+)\s*(?:->|=>|→|:)\s*([A-Za-z0-9_]+)"
    )

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self._classifier = None
        self.keyword_patterns = self._load_keyword_patterns()

    def _load_keyword_patterns(self) -> Dict[ProblemType, List[str]]:
        return {
            ProblemType.SORTING: ["sort", "order", "arrange", "organize", "rank"],
            ProblemType.PATHFINDING: ["path", "route", "navigate", "shortest", "distance", "graph"],
            ProblemType.OPTIMIZATION: ["optimize", "maximize", "minimize", "best", "knapsack", "allocate", "anneal", "genetic", "hill climbing", "ant colony", "particle swarm", "swarm"],
            ProblemType.ML: ["train", "model", "learn", "neural", "xgboost", "random_forest", "lightgbm", "gmm", "pca", "dimensionality", "reduce dimension", "t-sne", "tsne", "dimensionality reduction", "pca reduction"],
            ProblemType.NLP: ["text", "sentence", "semantic", "corpus", "language", "tokenize", "sentiment", "summariz", "enrich", "expand", "visualize word", "arithmetic", "rag", "retrieval", "retrieve", "analogy", "terminology", "vocabulary", "word2vec", "embedding"],
            ProblemType.COMPUTER_VISION: ["detect", "segment", "recognize", "object", "satellite"],
            ProblemType.IMAGE_PROCESSING: ["blur", "filter", "denoise", "edge", "image", "satellite", "template", "pattern", "morpholog", "segment"],
            ProblemType.CLUSTERING: ["cluster", "group", "partition", "kmeans", "dbscan"],
            ProblemType.CLASSIFICATION: ["classify", "categorize", "label", "predict class", "logistic_regression", "knn", "svm", "naive_bayes", "decision_tree", "risk tier"],
            ProblemType.REGRESSION: ["predict", "regress", "predict value", "forecast", "estimate", "linear_regression", "ridge", "lasso", "growth rate"],
            ProblemType.SEARCH: ["find", "search", "locate", "match", "retrieve"],
            ProblemType.GENERATION: ["generate", "create", "produce", "synthesize", "creative"],
            ProblemType.SCHEDULING: ["schedule", "timetable", "assign", "allocate time"],
        }

    def _get_classifier(self):
        """Load DistilBERT on first call. Cache on instance."""
        if self._classifier is None:
            try:
                from transformers import pipeline as hf_pipeline
                self._classifier = hf_pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=-1,
                )
                logger.info("Loaded zero-shot classifier for question parsing")
            except ImportError:
                logger.warning(
                    "transformers not installed. "
                    "Run: pip install aalgoi[transformer]"
                )
                self.use_transformer = False
                return None
            except Exception as e:
                logger.warning("Failed to load transformer: %s", e)
                self.use_transformer = False
                return None
        return self._classifier

    def _apply_shorthand(self, text: str) -> str:
        """Expand shorthand aliases into full phrases the parser handles."""
        text = text.strip().lower()

        match = self.ARROW_PATTERN.search(text)
        if match:
            return f"find shortest path from {match.group(1)} to {match.group(2)}"

        for short, full in sorted(self.SHORTHAND.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(short) + r'\b'
            if re.search(pattern, text):
                full_in_text = re.search(r'\b' + re.escape(full) + r'\b', text)
                if full_in_text:
                    continue
                return re.sub(pattern, full, text, count=1)

        return text

    def parse(self, question: str, data: Any = None) -> ProblemSpec:
        """
        Main entry point: question string -> ProblemSpec

        Args:
            question: Natural language question
            data: Optional data to infer type from

        Returns:
            ProblemSpec ready for UniversalSolver
        """
        question = self._apply_shorthand(question)
        question_lower = question.lower()

        problem_type = self._detect_problem_type(question_lower)
        constraints = self._extract_constraints(question_lower)
        objectives = self._extract_objectives(question_lower)
        params = self._extract_parameters(question_lower, problem_type)

        spec = ProblemSpec(
            name=self._generate_name(question),
            problem_type=problem_type,
            description=question,
            constraints=[Constraint(c) if isinstance(c, str) else c for c in constraints],
            objectives=[Objective(**o) if isinstance(o, dict) else o for o in objectives],
            inputs=params,
        )

        if data is not None:
            spec = self._enrich_from_data(spec, data)

        return spec

    def _detect_problem_type(self, question: str) -> ProblemType:
        # Step 1: Keywords (zero cost, always runs first)
        scores = {}
        for ptype, keywords in self.keyword_patterns.items():
            score = sum(1 for kw in keywords if kw in question)
            if score > 0:
                scores[ptype] = score

        if scores:
            return max(scores, key=scores.get)

        # Step 2: Transformer (only if keywords found nothing)
        if self.use_transformer:
            clf = self._get_classifier()
            if clf is not None:
                candidate_labels = [pt.value for pt in ProblemType if pt != ProblemType.UNKNOWN]
                try:
                    result = clf(question, candidate_labels)
                    if result["scores"][0] > 0.5:
                        return ProblemType(result["labels"][0])
                except Exception:
                    pass

        return ProblemType.UNKNOWN

    def _extract_constraints(self, question: str) -> List[str]:
        constraints = []

        time_match = re.search(r"under (\d+\.?\d*)\s*(second|ms|millisecond)", question)
        if time_match:
            time_val = float(time_match.group(1))
            unit = time_match.group(2)
            time_ms = time_val * 1000 if "second" in unit else time_val
            constraints.append(f"time_budget_ms <= {time_ms}")

        memory_match = re.search(r"(\d+)\s*(mb|gb)\s*memory", question)
        if memory_match:
            constraints.append(f"memory_budget_mb <= {memory_match.group(1)}")

        return constraints

    def _extract_objectives(self, question: str) -> List[Dict]:
        objectives = []

        if re.search(r"maximize\s+\w+", question):
            match = re.search(r"maximize\s+(\w+)", question)
            objectives.append({"direction": "maximize", "metric": match.group(1)})

        if re.search(r"minimize\s+\w+", question):
            match = re.search(r"minimize\s+(\w+)", question)
            objectives.append({"direction": "minimize", "metric": match.group(1)})

        if "shortest" in question or "fastest" in question:
            objectives.append({"direction": "minimize", "metric": "total_time"})

        if any(w in question for w in ["fast", "quick", "rapid"]):
            objectives.append({"direction": "minimize", "metric": "execution_time"})

        return objectives

    def _extract_parameters(self, question: str, problem_type: ProblemType) -> Dict:
        params = {}

        if problem_type in (ProblemType.ML, ProblemType.NLP):
            dim_match = re.search(r"(\d+)\s*(?:dimension|dim|d)\b", question)
            if dim_match:
                params["vector_size"] = int(dim_match.group(1))

            domain_match = re.search(r"(medical|legal|finance|tech|general)\s+(?:corpus|domain|text)", question)
            if domain_match:
                params["domain"] = domain_match.group(1)

            epoch_match = re.search(r"(\d+)\s+epoch", question)
            if epoch_match:
                params["epochs"] = int(epoch_match.group(1))

            topn_match = re.search(r"(\d+)\s+(?:word|similar|result)", question)
            if topn_match:
                params["topn"] = int(topn_match.group(1))

            word_match = re.search(r"word[:\s]+\"([^\"]+)\"", question)
            if not word_match:
                word_match = re.search(r"for\s+(\w+)", question)
            if word_match:
                params["input_word"] = word_match.group(1)

        return params

    def _generate_name(self, question: str) -> str:
        words = re.findall(r"\w+", question.lower())[:5]
        return "_".join(words) if words else "unknown_problem"

    def _enrich_from_data(self, spec: ProblemSpec, data: Any) -> ProblemSpec:
        if isinstance(data, dict):
            if "corpus" in data or "sentences" in data:
                spec.problem_type = ProblemType.ML
            elif "graph" in data:
                spec.problem_type = ProblemType.PATHFINDING
            elif "items" in data and "capacity" in data:
                spec.problem_type = ProblemType.OPTIMIZATION
            elif "embeddings" in data:
                spec.problem_type = ProblemType.NLP
            elif "model" in data:
                spec.problem_type = ProblemType.ML
            elif "X_train" in data:
                y = data.get("y_train")
                if y is not None:
                    import numpy as _np
                    if isinstance(y, (list, tuple, _np.ndarray)):
                        y_arr = _np.asarray(y)
                        if y_arr.ndim == 1:
                            if y_arr.dtype.kind in 'UO':
                                spec.problem_type = ProblemType.CLASSIFICATION
                            elif y_arr.dtype.kind in 'biuf':
                                unique_count = len(_np.unique(y_arr))
                                if unique_count < len(y_arr) * 0.5 and unique_count < 20:
                                    spec.problem_type = ProblemType.CLASSIFICATION
                                else:
                                    spec.problem_type = ProblemType.REGRESSION
                            else:
                                spec.problem_type = ProblemType.CLUSTERING
                        else:
                            spec.problem_type = ProblemType.CLUSTERING
                else:
                    spec.problem_type = ProblemType.CLUSTERING
        return spec
