"""Tests for UniversalMetaController and MetaController pure methods."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aalgoi.algorithms.base import Algorithm
from aalgoi.core.meta_controller import MetaController, UniversalMetaController
from aalgoi.core.problem_spec import ProblemSpec, ProblemType


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

class DummyAlgo(Algorithm):
    """Algorithm with configurable metadata for testing."""

    def __init__(self, name: str = "dummy", tags=None, best_for=None,
                 patterns=None, problem_types=None):
        super().__init__()
        self.name = name
        self.tags = tags or []
        self.best_for = best_for or []
        self.patterns = patterns or []
        self.problem_types = problem_types or []
        self.time_complexity = "O(1)"
        self.space_complexity = "O(1)"

    def process(self, data):
        return data


def make_registry(names: list[str]) -> dict[str, Algorithm]:
    return {n: DummyAlgo(name=n) for n in names}


# ==============================
#  UniversalMetaController tests
# ==============================

class TestBuildFallbackChain:
    def test_returns_ordered_list(self):
        uc = UniversalMetaController()
        chain = uc._build_fallback_chain()
        assert chain == ["identity", "safe_sort", "quicksort", "bfs_path", "greedy_knapsack"]

    def test_always_same_result(self):
        a = UniversalMetaController()._build_fallback_chain()
        b = UniversalMetaController()._build_fallback_chain()
        assert a == b

    def test_static_identity(self):
        assert UniversalMetaController._build_fallback_chain(
            UniversalMetaController()
        ) == ["identity", "safe_sort", "quicksort", "bfs_path", "greedy_knapsack"]


class TestGetFallbackAlgorithms:
    def test_all_in_registry(self):
        names = ["identity", "safe_sort", "quicksort", "bfs_path", "greedy_knapsack"]
        reg = make_registry(names)
        uc = UniversalMetaController(algorithm_registry=reg)
        result = uc._get_fallback_algorithms()
        assert len(result) == 5
        assert [a.name for a in result] == names

    def test_partial_in_registry(self):
        reg = make_registry(["identity", "quicksort"])
        uc = UniversalMetaController(algorithm_registry=reg)
        result = uc._get_fallback_algorithms()
        assert len(result) == 2
        assert result[0].name == "identity"
        assert result[1].name == "quicksort"

    def test_none_in_registry(self):
        reg = make_registry(["something_else"])
        uc = UniversalMetaController(algorithm_registry=reg)
        assert uc._get_fallback_algorithms() == []

    def test_empty_registry(self):
        uc = UniversalMetaController(algorithm_registry={})
        assert uc._get_fallback_algorithms() == []


class TestSplitWords:
    @staticmethod
    def test_camel_case():
        result = UniversalMetaController._split_words("camelCase")
        assert result == {"camel", "case"}

    @staticmethod
    def test_snake_case():
        result = UniversalMetaController._split_words("snake_case")
        assert result == {"snake", "case"}

    @staticmethod
    def test_pascal_case():
        result = UniversalMetaController._split_words("PascalCase")
        assert result == {"pascal", "case"}

    @staticmethod
    def test_special_chars():
        result = UniversalMetaController._split_words("hello-world!foo")
        assert result == {"hello", "world", "foo"}

    @staticmethod
    def test_mixed():
        result = UniversalMetaController._split_words("kMeansClustering-v2")
        assert result == {"k", "means", "clustering", "v2"}

    @staticmethod
    def test_empty_string():
        assert UniversalMetaController._split_words("") == set()

    @staticmethod
    def test_only_special():
        assert UniversalMetaController._split_words("!@#$%") == set()

    @staticmethod
    def test_numbers():
        result = UniversalMetaController._split_words("v1_2test")
        assert result == {"v1", "2test"}


class TestActionWords:
    @staticmethod
    def test_returns_dict_with_expected_keys():
        aw = UniversalMetaController._action_words()
        expected_keys = [
            "sort", "path", "classify", "regress", "cluster", "reduce",
            "sentiment", "summarize", "retrieve", "search", "enrich",
            "arithmetic", "visualize", "expand", "generate", "train",
            "edge", "segment", "template", "morphological", "knapsack",
            "annealing", "genetic", "hill", "pso", "aco", "greedy",
        ]
        for k in expected_keys:
            assert k in aw, f"Missing key: {k}"
        assert len(aw) == len(expected_keys)

    @staticmethod
    def test_values_are_sets_of_strings():
        for k, v in UniversalMetaController._action_words().items():
            assert isinstance(v, set)
            for item in v:
                assert isinstance(item, str)

    @staticmethod
    def test_sort_action():
        aw = UniversalMetaController._action_words()
        assert "sort" in aw["sort"]
        assert "sorting" in aw["sort"]

    @staticmethod
    def test_deterministic():
        a = UniversalMetaController._action_words()
        b = UniversalMetaController._action_words()
        assert a == b


class TestScoreAlgorithmForQuery:
    def test_exact_name_match_scores_high(self):
        reg = make_registry(["quicksort"])
        uc = UniversalMetaController(algorithm_registry=reg)
        score = uc._score_algorithm_for_query("quicksort", "sort my data using quicksort")
        assert score > 10.0

    def test_no_match_returns_non_negative_or_small(self):
        reg = make_registry(["quicksort"])
        uc = UniversalMetaController(algorithm_registry=reg)
        score = uc._score_algorithm_for_query("quicksort", "hello world")
        # may be negative due to name penalty
        assert score <= 0.0

    def test_algo_not_in_registry_returns_zero(self):
        uc = UniversalMetaController()
        assert uc._score_algorithm_for_query("nonexistent", "query") == 0.0

    def test_tag_match_adds_points(self):
        algo = DummyAlgo(name="my_algo", tags=["sorting", "fast"])
        reg = {"my_algo": algo}
        uc = UniversalMetaController(algorithm_registry=reg)
        score = uc._score_algorithm_for_query("my_algo", "sorting problem")
        # tag "sorting" found in query -> +5
        # also best_for and patterns empty; name "my_algo" not in query -> -5
        # net roughly 0
        assert score >= 0.0

    def test_best_for_match_adds_points(self):
        algo = DummyAlgo(name="my_algo", best_for=["random data", "large_n"])
        reg = {"my_algo": algo}
        uc = UniversalMetaController(algorithm_registry=reg)
        score = uc._score_algorithm_for_query("my_algo", "large random data")
        # "random data" -> {"random", "data"} -> overlap with {"large","random","data"} = 2 * 8 = 16
        # "large_n" -> {"large", "n"} -> overlap = {"large"} = 1 * 8 = 8
        # total best_for = 24, name penalty = -5
        assert score == pytest.approx(19.0, abs=0.1)

    def test_pattern_match_adds_points(self):
        algo = DummyAlgo(name="my_algo", patterns=["conquer"])
        reg = {"my_algo": algo}
        uc = UniversalMetaController(algorithm_registry=reg)
        score = uc._score_algorithm_for_query("my_algo", "divide and conquer")
        expected = 3.0 - 5.0  # pattern matched (+3), no name match (-5)
        assert score == pytest.approx(expected, abs=0.1)

    def test_action_penalty_applied(self):
        algo = DummyAlgo(name="sort_algo", tags=["sort"])
        reg = {"sort_algo": algo}
        uc = UniversalMetaController(algorithm_registry=reg)
        # query wants path action but algo only has sort
        score = uc._score_algorithm_for_query("sort_algo", "find shortest path")
        # name penalty -5, no tag match, action mismatch -20 => score = -25
        assert score < 0.0


class TestExtractKgConstraints:
    def test_sorting_nearly_sorted(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "NearlySorted" in c

    def test_sorting_small_n(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        data = [5, 3, 1]
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "SmallN" in c

    def test_sorting_large_n(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        data = list(range(20000))
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "LargeN" in c
        assert "NearlySorted" in c

    def test_sorting_empty(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, [])
        assert c == []

    def test_pathfinding_weighted(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        data = {"graph": {"A": {"B": 5}, "B": {}}, "start": "A"}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "WeightedGraph" in c
        assert "PointToPoint" in c

    def test_pathfinding_unweighted(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        data = {"graph": {"A": {"B": {}}, "B": {}}}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "UnweightedGraph" in c

    def test_pathfinding_no_graph_key(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
        data = {"start": "A"}  # data is dict but no "graph" key
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "UnweightedGraph" in c
        assert "PointToPoint" in c

    def test_optimization_resource_allocation(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        data = {"items": [{"value": 10, "weight": 5}]}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "ResourceAllocation" in c

    def test_optimization_no_items(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.OPTIMIZATION)
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, {})
        assert c == []

    def test_classification_high_dim(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.CLASSIFICATION)
        X = np.random.randn(50, 200)
        data = {"X_train": X}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "HighDim" in c
        assert "SmallN" in c

    def test_ml_very_high_dim(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.ML)
        X = np.random.randn(50, 2000)
        data = {"X_train": X}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "HighDim" in c
        assert "VeryHighDim" in c
        assert "SmallN" in c

    def test_ml_large_n(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.ML)
        X = np.random.randn(50000, 10)
        data = {"X_train": X}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "LargeN" in c

    def test_ml_very_large_n(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.ML)
        X = np.random.randn(200000, 10)
        data = {"X_train": X}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "LargeN" in c
        assert "VeryLargeN" in c

    def test_regression_scipy_skip_on_import_error(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.REGRESSION)
        X = np.random.randn(100, 50)
        data = {"X_train": X}
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        # Should not crash even without scipy
        assert isinstance(c, list)

    def test_string_data_sorting(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.SORTING)
        data = ["z", "a", "m"]
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert isinstance(c, list)

    def test_transformation_small(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.TRANSFORMATION)
        data = [42, 43]
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "SmallN" in c

    def test_unknown_problem_type(self):
        spec = ProblemSpec(name="test", problem_type=ProblemType.UNKNOWN)
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, {})
        assert c == []

    def test_problem_type_value_override(self):
        # Test the fallback path where problem_type may not be a ProblemType enum
        from types import SimpleNamespace
        spec = SimpleNamespace()
        spec.problem_type = ProblemType.SORTING
        data = [1, 2, 3, 4, 5]
        uc = UniversalMetaController()
        c = uc._extract_kg_constraints(spec, data)
        assert "NearlySorted" in c
        assert "SmallN" in c


class TestGetLastSelectionAndConfidence:
    def test_defaults(self):
        uc = UniversalMetaController()
        assert uc.get_last_selection() is None
        assert uc.get_last_confidence() == 0.0

    def test_after_setting_values(self):
        uc = UniversalMetaController()
        uc._last_selection = {"algorithm": "quicksort"}
        uc._last_confidence = 0.85
        assert uc.get_last_selection() == {"algorithm": "quicksort"}
        assert uc.get_last_confidence() == 0.85


# ========================
#  MetaController tests
# ========================

class TestComputeKbConfidence:
    def test_empty_list_returns_default(self):
        mc = MetaController(make_registry(["a"]))
        assert mc._compute_kb_confidence([]) == 0.2

    def test_fewer_than_3_returns_default(self):
        mc = MetaController(make_registry(["a"]))
        assert mc._compute_kb_confidence([{}, {}]) == 0.2

    def test_no_scores_returns_default(self):
        mc = MetaController(make_registry(["a"]))
        records = [{"metrics": {}}, {"metrics": {}}, {"metrics": {}}]
        assert mc._compute_kb_confidence(records) == 0.2

    def test_average_of_scores(self):
        mc = MetaController(make_registry(["a"]))
        records = [
            {"metrics": {"score": 0.8}},
            {"metrics": {"score": 0.6}},
            {"metrics": {"score": 0.4}},
        ]
        result = mc._compute_kb_confidence(records)
        assert result == pytest.approx(0.6)

    def test_clamps_to_01(self):
        mc = MetaController(make_registry(["a"]))
        records = [
            {"metrics": {"score": 2.0}},
            {"metrics": {"score": 3.0}},
            {"metrics": {"score": 4.0}},
        ]
        assert mc._compute_kb_confidence(records) == 1.0


class TestRuleBasedSelectSingle:
    def test_small_data_returns_insertion_sort(self):
        reg = make_registry(["insertion_sort", "timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 5, "type": "list", "patterns": {}},
            "features": {},
            "constraints": {},
        }
        result = mc._rule_based_select_single(ctx)
        assert result == "insertion_sort"

    def test_sorted_data_returns_timsort(self):
        reg = make_registry(["insertion_sort", "timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 100, "type": "list",
                             "patterns": {"is_sorted": True}},
            "features": {},
            "constraints": {},
        }
        assert mc._rule_based_select_single(ctx) == "timsort"

    def test_nearly_sorted_large_returns_timsort(self):
        reg = make_registry(["insertion_sort", "timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 1000, "type": "list",
                             "patterns": {"is_nearly_sorted": True}},
            "features": {},
            "constraints": {},
        }
        assert mc._rule_based_select_single(ctx) == "timsort"

    def test_large_data_returns_timsort(self):
        reg = make_registry(["insertion_sort", "timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 200000, "type": "list",
                             "patterns": {}, "statistics": {}},
            "features": {},
            "constraints": {},
        }
        assert mc._rule_based_select_single(ctx) == "timsort"

    def test_large_uniform_data_radix_if_available(self):
        reg = make_registry(["radix_sort", "timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {
                "size": 200000, "type": "list",
                "patterns": {},
                "statistics": {"is_uniform": True, "unique_ratio": 0.05},
            },
            "features": {},
            "constraints": {},
        }
        assert mc._rule_based_select_single(ctx) == "radix_sort"

    def test_memory_constrained_large_returns_heap_sort(self):
        reg = make_registry(["heap_sort", "timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 50000, "type": "list", "patterns": {}},
            "features": {"mem_free_ratio": 0.1},
            "constraints": {},
        }
        assert mc._rule_based_select_single(ctx) == "heap_sort"

    def test_fast_time_budget_returns_timsort(self):
        reg = make_registry(["timsort", "quicksort", "insertion_sort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 5000, "type": "list", "patterns": {}},
            "features": {},
            "constraints": {"time_budget_ms": 30},
        }
        assert mc._rule_based_select_single(ctx) == "timsort"

    def test_non_list_data_type_falls_back(self):
        reg = make_registry(["timsort", "quicksort"])
        mc = MetaController(reg)
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 100, "type": "dict", "patterns": {}},
            "features": {},
            "constraints": {},
        }
        result = mc._rule_based_select_single(ctx)
        assert result in ("timsort", "quicksort")

    def test_empty_registry_still_returns_domain_algo(self):
        mc = MetaController({})
        ctx = {"task_type": "sorting", "data_profile": {}}
        result = mc._rule_based_select_single(ctx)
        # domain_map is hardcoded; registry is not used in this branch
        assert result == "quicksort"

    def test_domain_algo_name_check_delegates_to_map_not_registry(self):
        reg = make_registry(["quicksort", "merge_sort"])
        mc = MetaController(reg)
        # _rule_based_select_single checks domain_algos (from hardcoded map),
        # not the registry, for "insertion_sort" availability
        ctx = {
            "task_type": "sorting",
            "data_profile": {"size": 5, "type": "list", "patterns": {}},
            "features": {},
            "constraints": {},
        }
        result = mc._rule_based_select_single(ctx)
        assert result == "insertion_sort"


class TestVectorize:
    def test_all_features_present(self):
        mc = MetaController(make_registry(["a"]))
        features = {
            "data_size_log": 1.0,
            "is_numeric": 1.0,
            "is_sorted": 1.0,
            "is_nearly_sorted": 1.0,
            "cpu_free": 1.0,
            "mem_free_ratio": 1.0,
            "cpu_count": 1.0,
            "time_budget_norm": 1.0,
            "priority_speed": 1.0,
            "priority_accuracy": 1.0,
        }
        vec = mc._vectorize(features)
        assert vec == [1.0] * 10

    def test_missing_features_default_to_zero(self):
        mc = MetaController(make_registry(["a"]))
        vec = mc._vectorize({})
        assert vec == [0.0] * 10

    def test_partial_features(self):
        mc = MetaController(make_registry(["a"]))
        vec = mc._vectorize({"data_size_log": 2.5, "is_sorted": 1.0})
        assert vec[0] == 2.5
        assert vec[1] == 0.0
        assert vec[2] == 1.0
        assert vec[3:] == [0.0] * 7

    def test_consistent_length(self):
        mc = MetaController(make_registry(["a"]))
        assert len(mc._vectorize({})) == 10
        assert len(mc._vectorize({"data_size_log": 1})) == 10


class TestComputeSimilarity:
    def test_identical_contexts(self):
        mc = MetaController(make_registry(["a"]))
        ctx = {"features": {"data_size_log": 1.0, "is_numeric": 1.0}}
        sim = mc._compute_similarity(ctx, ctx)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_contexts(self):
        mc = MetaController(make_registry(["a"]))
        ctx_a = {"features": {"data_size_log": 1.0, "is_numeric": 0.0}}
        ctx_b = {"features": {"data_size_log": 0.0, "is_numeric": 1.0}}
        sim = mc._compute_similarity(ctx_a, ctx_b)
        assert sim == pytest.approx(0.0)

    def test_partial_similarity(self):
        mc = MetaController(make_registry(["a"]))
        ctx_a = {"features": {"data_size_log": 1.0, "is_sorted": 1.0, "cpu_free": 0.0}}
        ctx_b = {"features": {"data_size_log": 1.0, "is_sorted": 0.0, "cpu_free": 0.0}}
        dot = 1.0 * 1.0
        norm_a = math.sqrt(1.0**2 + 1.0**2 + 0.0**2)
        norm_b = math.sqrt(1.0**2 + 0.0**2 + 0.0**2)
        expected = dot / (norm_a * norm_b)
        sim = mc._compute_similarity(ctx_a, ctx_b)
        assert sim == pytest.approx(expected)

    def test_zero_vector_returns_zero(self):
        mc = MetaController(make_registry(["a"]))
        ctx_a = {"features": {}}
        ctx_b = {"features": {}}
        assert mc._compute_similarity(ctx_a, ctx_b) == 0.0

    def test_with_no_feature_key(self):
        mc = MetaController(make_registry(["a"]))
        ctx_a = {}
        ctx_b = {"features": {"data_size_log": 1.0}}
        # still works because _vectorize handles missing features
        sim = mc._compute_similarity(ctx_a, ctx_b)
        assert isinstance(sim, float)


class TestFindSimilarContexts:
    def test_empty_history(self):
        mc = MetaController(make_registry(["a"]))
        assert mc._find_similar_contexts({"features": {}}) == []

    def test_single_history_record(self):
        mc = MetaController(make_registry(["a"]))
        mc.history = [{"context": {"features": {"data_size_log": 1.0}}}]
        result = mc._find_similar_contexts(
            {"features": {"data_size_log": 1.0}}, top_k=5
        )
        assert len(result) == 1

    def test_returns_top_k(self):
        mc = MetaController(make_registry(["a"]))
        mc.history = [
            {"context": {"features": {"data_size_log": float(i)}}}
            for i in range(20)
        ]
        result = mc._find_similar_contexts(
            {"features": {"data_size_log": 0.0}}, top_k=3
        )
        assert len(result) == 3

    def test_orders_by_similarity_descending(self):
        mc = MetaController(make_registry(["a"]))
        mc.history = [
            {"context": {"features": {"data_size_log": 0.0}}},
            {"context": {"features": {"data_size_log": 10.0}}},
            {"context": {"features": {"data_size_log": 1.0}}},
        ]
        result = mc._find_similar_contexts(
            {"features": {"data_size_log": 0.0}}, top_k=3
        )
        # first should be most similar (data_size_log=0.0)
        assert result[0]["context"]["features"]["data_size_log"] == 0.0


class TestSelectFromHistoryWeighted:
    def test_empty_history_returns_none(self):
        mc = MetaController(make_registry(["a"]))
        assert mc._select_from_history_weighted([], {}) is None

    def test_selects_highest_weighted_avg(self):
        mc = MetaController(make_registry(["a", "b"]))
        ctx = {"features": {"data_size_log": 1.0}}
        similar = [
            {"algorithms": ["a"], "score": 0.9, "context": ctx},
            {"algorithms": ["b"], "score": 0.5, "context": ctx},
        ]
        result = mc._select_from_history_weighted(similar, ctx)
        assert result == "a"

    def test_similarity_weighting(self):
        mc = MetaController(make_registry(["a", "b"]))
        # Vectors pointing in different directions so cosine sim differs
        similar_ctx = {"features": {"data_size_log": 1.0, "is_numeric": 1.0}}
        dissimilar_ctx = {"features": {"data_size_log": 0.0, "is_numeric": 0.0,
                                       "cpu_free": 1.0}}
        similar = [
            {"algorithms": ["a"], "score": 1.0, "context": dissimilar_ctx},
            {"algorithms": ["b"], "score": 0.6, "context": similar_ctx},
        ]
        # b has lower raw score but higher similarity to current context
        current_ctx = {"features": {"data_size_log": 1.0, "is_numeric": 1.0}}
        result = mc._select_from_history_weighted(similar, current_ctx)
        assert result == "b"

    def test_no_algorithms_key_returns_none(self):
        mc = MetaController(make_registry(["a"]))
        similar = [{"score": 0.5, "context": {"features": {}}}]
        assert mc._select_from_history_weighted(similar, {}) is None


class TestGetAverageScore:
    def test_no_history_returns_default(self):
        mc = MetaController(make_registry(["a"]))
        assert mc._get_average_score("a") == 0.5

    def test_single_record(self):
        mc = MetaController(make_registry(["a"]))
        mc.history = [{"algorithms": ["a"], "score": 0.8}]
        assert mc._get_average_score("a") == pytest.approx(0.8)

    def test_multiple_records_averaged(self):
        mc = MetaController(make_registry(["a"]))
        mc.history = [
            {"algorithms": ["a"], "score": 1.0},
            {"algorithms": ["a"], "score": 0.0},
            {"algorithms": ["b"], "score": 0.5},
        ]
        assert mc._get_average_score("a") == pytest.approx(0.5)

    def test_algo_not_found_returns_default(self):
        mc = MetaController(make_registry(["a"]))
        mc.history = [{"algorithms": ["a"], "score": 0.9}]
        assert mc._get_average_score("nonexistent") == 0.5


class TestGetDomainAlgorithms:
    def test_sorting_returns_sorting_algos(self):
        mc = MetaController(make_registry(["quicksort", "insertion_sort"]))
        result = mc._get_domain_algorithms({"task_type": "sorting"})
        assert "quicksort" in result
        assert "insertion_sort" in result

    def test_unknown_task_type_falls_back_to_registry(self):
        reg = make_registry(["a", "b"])
        mc = MetaController(reg)
        result = mc._get_domain_algorithms({"task_type": "unknown_type"})
        assert result == list(reg.keys())

    def test_auto_task_with_list_data_returns_sorting(self):
        mc = MetaController(make_registry(["quicksort", "insertion_sort"]))
        result = mc._get_domain_algorithms({
            "task_type": "auto",
            "data_profile": {"type": "list"},
        })
        assert "quicksort" in result
        assert "insertion_sort" in result

    def test_auto_task_with_ndarray_returns_image_processing(self):
        mc = MetaController(make_registry(["gaussian_blur", "sobel_edge", "quicksort"]))
        result = mc._get_domain_algorithms({
            "task_type": "auto",
            "data_profile": {"type": "ndarray"},
        })
        assert "gaussian_blur" in result
        assert "sobel_edge" in result
        assert "quicksort" not in result

    def test_auto_task_with_unknown_type_falls_to_sorting(self):
        mc = MetaController(make_registry(["quicksort", "insertion_sort"]))
        result = mc._get_domain_algorithms({
            "task_type": "",
            "data_profile": {"type": "str"},
        })
        assert "quicksort" in result

    def test_ml_domain(self):
        mc = MetaController(make_registry(["kmeans", "linear_regression"]))
        result = mc._get_domain_algorithms({"task_type": "ml"})
        assert "kmeans" in result
        assert "linear_regression" in result


class TestGetStats:
    def test_returns_dict_with_expected_keys(self):
        mc = MetaController(make_registry(["a", "b"]))
        stats = mc.get_stats()
        assert "strategy" in stats
        assert "history_size" in stats
        assert "trained" in stats
        assert "generation" in stats
        assert "last_confidence" in stats
        assert "last_reason" in stats
        assert "bandit" in stats
        assert "llm" in stats
        assert "algorithm_performance" in stats

    def test_strategy_reflects_init(self):
        mc = MetaController(make_registry(["a"]), strategy="rule-based")
        assert mc.get_stats()["strategy"] == "rule-based"

    def test_history_size(self):
        mc = MetaController(make_registry(["a"]))
        assert mc.get_stats()["history_size"] == 0
        mc.history = [{"algorithms": ["a"], "score": 0.5, "metrics": {}}]
        assert mc.get_stats()["history_size"] == 1

    def test_trained_state(self):
        mc = MetaController(make_registry(["a"]))
        assert mc.get_stats()["trained"] is False
        mc._trained = True
        assert mc.get_stats()["trained"] is True

    def test_generation(self):
        mc = MetaController(make_registry(["a"]))
        assert mc.get_stats()["generation"] == 0
        mc._generation = 5
        assert mc.get_stats()["generation"] == 5

    def test_last_confidence_and_reason(self):
        mc = MetaController(make_registry(["a"]))
        assert mc.get_stats()["last_confidence"] == 0.0
        assert mc.get_stats()["last_reason"] == ""
        mc._last_confidence = 0.9
        mc._last_reason = "test"
        assert mc.get_stats()["last_confidence"] == 0.9
        assert mc.get_stats()["last_reason"] == "test"

    def test_algorithm_performance_empty(self):
        mc = MetaController(make_registry(["a"]))
        assert mc.get_stats()["algorithm_performance"] == {}

    def test_algorithm_performance_with_data(self):
        mc = MetaController(make_registry(["a"]))
        mc.knowledge_base["a"] = [{"score": 0.8}, {"score": 0.6}]
        perf = mc.get_stats()["algorithm_performance"]
        assert "a" in perf
        assert perf["a"]["count"] == 2
        assert perf["a"]["avg_score"] == pytest.approx(0.7)

    def test_bandit_stats_present(self):
        mc = MetaController(make_registry(["a"]))
        stats = mc.get_stats()
        assert isinstance(stats["bandit"], dict)

    def test_llm_stats_present(self):
        mc = MetaController(make_registry(["a"]))
        stats = mc.get_stats()
        assert isinstance(stats["llm"], dict)
