"""Tests for algorithm solvers across all domains."""
import pytest
from importlib.util import find_spec
import tempfile

from aalgoi._core import Mind, _rule_based_solve

_have_sklearn = find_spec("sklearn") is not None
_have_scipy = find_spec("scipy") is not None


class TestSortingSolver:
    def test_sort_ascending(self):
        result = _rule_based_solve("sort the array", [5, 3, 8, 1, 9, 2, 7, 4, 6])
        assert result.output == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert result.algorithm == "tim_sort"

    def test_sort_descending(self):
        result = _rule_based_solve("sort descending", [5, 3, 8, 1, 9, 2])
        assert result.output == [9, 8, 5, 3, 2, 1]

    def test_sort_organize(self):
        result = _rule_based_solve("organize these numbers", [3, 1, 2])
        assert result.output == [1, 2, 3]

    def test_sort_order(self):
        result = _rule_based_solve("order the list", [3, 1, 2])
        assert result.output == [1, 2, 3]

    def test_sort_with_empty_list(self):
        result = _rule_based_solve("sort", [])
        assert result is None or result.output == []


class TestSearchSolver:
    def test_binary_search_sorted(self):
        result = _rule_based_solve("binary search for 5", {"nums": [1, 3, 5, 7, 9], "target": 5})
        assert result.output == 2
        assert result.algorithm == "binary_search"
        assert result.complexity == "O(log n)"

    def test_binary_search_unsorted(self):
        result = _rule_based_solve("binary search for 5", {"nums": [5, 3, 1, 7, 2], "target": 5})
        assert result.algorithm == "linear_search"
        assert result.complexity == "O(n)"

    def test_search_not_found(self):
        result = _rule_based_solve("binary search for 99", {"nums": [1, 3, 5, 7], "target": 99})
        # Not found returns None from rule-based solver
        assert result is None or result.output == -1

    def test_linear_search(self):
        result = _rule_based_solve("find index of 3", {"nums": [1, 2, 3, 4, 5], "target": 3})
        assert result.output == 2
        assert result.algorithm in ["linear_search", "binary_search"]


class TestTwoSumSolver:
    def test_two_sum_basic(self):
        result = _rule_based_solve("two sum", {"nums": [2, 7, 11, 15], "target": 9})
        assert result.output == [0, 1]
        assert result.algorithm == "hash_complement"
        assert result.complexity == "O(n)"

    def test_two_sum_multiple_pairs(self):
        result = _rule_based_solve("two sum", {"nums": [3, 2, 4], "target": 6})
        assert result.output == [1, 2]

    def test_two_sum_no_pair(self):
        result = _rule_based_solve("two sum", {"nums": [1, 2, 3], "target": 10})
        assert result is None


class TestKadaneSolver:
    def test_maximum_subarray_basic(self):
        result = _rule_based_solve("maximum subarray", [-2, 1, -3, 4, -1, 2, 1, -5, 4])
        assert result.output == 6
        assert result.algorithm == "kadane"
        assert result.complexity == "O(n)"

    def test_maximum_subarray_all_negative(self):
        result = _rule_based_solve("maximum subarray", [-5, -2, -9, -1])
        assert result.output == -1

    def test_maximum_subarray_single_element(self):
        result = _rule_based_solve("maximum subarray", [5])
        assert result.output == 5

    def test_kadane_keyword(self):
        result = _rule_based_solve("kadane algorithm", [-2, 1, -3, 4])
        assert result.output == 4
        assert result.algorithm == "kadane"


class TestGCDLCMSolver:
    def test_gcd_basic(self):
        result = _rule_based_solve("gcd", {"a": 12, "b": 8})
        assert result.output == 4
        assert result.algorithm == "euclidean_gcd"

    def test_gcd_greatest_common(self):
        result = _rule_based_solve("greatest common divisor", {"a": 48, "b": 18})
        assert result.output == 6

    def test_lcm_basic(self):
        result = _rule_based_solve("lcm", {"a": 4, "b": 6})
        assert result.output == 12
        assert result.algorithm == "lcm_via_gcd"

    def test_lcm_least_common(self):
        result = _rule_based_solve("least common multiple", {"a": 3, "b": 5})
        assert result.output == 15


class TestPathfindingSolver:
    def test_path_basic(self):
        graph = {"A": {"B": 1, "C": 4}, "B": {"C": 2}, "C": {}}
        result = _rule_based_solve("shortest path from A to C", {
            "graph": graph, "start": "A", "end": "C"
        })
        assert result is not None
        assert result.output is not None
        assert result.output[0] == "A"
        assert result.output[-1] == "C"

    def test_path_route(self):
        graph = {"S": {"A": 1, "B": 2}, "A": {"T": 3}, "B": {"T": 1}, "T": {}}
        result = _rule_based_solve("route from S to T", {
            "graph": graph, "start": "S", "end": "T"
        })
        assert result.output[0] == "S"
        assert result.output[-1] == "T"

    def test_path_navigate(self):
        graph = {"A": {"B": 1}, "B": {"C": 1}, "C": {}}
        result = _rule_based_solve("navigate from A to C", {
            "graph": graph, "start": "A", "end": "C"
        })
        assert result.output == ["A", "B", "C"]

    def test_path_no_networkx(self):
        graph = {"A": ["B", "C"], "B": ["C"], "C": []}
        result = _rule_based_solve("shortest path", {
            "graph": graph, "start": "A", "end": "C"
        })
        assert result is not None


class TestKnapsackSolver:
    def test_knapsack_basic(self):
        items = [
            {"name": "a", "weight": 2, "value": 3},
            {"name": "b", "weight": 3, "value": 4},
            {"name": "c", "weight": 4, "value": 5},
        ]
        result = _rule_based_solve("knapsack", {"items": items, "capacity": 5})
        assert result is not None
        assert result.output is not None

    def test_knapsack_maximize_value(self):
        items = [
            {"name": "a", "weight": 1, "value": 10},
            {"name": "b", "weight": 3, "value": 40},
            {"name": "c", "weight": 4, "value": 50},
        ]
        result = _rule_based_solve("knapsack maximize value", {"items": items, "capacity": 4})
        assert result is not None

    def test_knapsack_capacity(self):
        items = [{"name": "x", "weight": 5, "value": 10}]
        result = _rule_based_solve("capacity", {"items": items, "capacity": 3})
        assert result is not None


class TestSchedulingSolver:
    def test_schedule_deadline(self):
        tasks = [
            {"name": "task1", "deadline": 5},
            {"name": "task2", "deadline": 3},
            {"name": "task3", "deadline": 8},
        ]
        result = _rule_based_solve("schedule tasks", tasks)
        assert result is not None
        deadlines = [t.get("deadline", 0) for t in result.output]
        assert deadlines == sorted(deadlines)

    def test_schedule_timetable(self):
        tasks = [{"due": 10}, {"due": 5}]
        result = _rule_based_solve("timetable", tasks)
        assert result is not None

    def test_schedule_assign(self):
        tasks = [{"time": 3}, {"time": 1}]
        result = _rule_based_solve("assign tasks", tasks)
        assert result is not None


@pytest.mark.skipif(not _have_sklearn, reason="sklearn not installed")
class TestClusteringSolver:
    def test_cluster_basic(self):
        data = [[1, 1], [1.2, 1.1], [5, 5], [5.1, 5.2]]
        result = _rule_based_solve("cluster data", data)
        assert result is not None
        assert "labels" in result.output
        assert "centers" in result.output

    def test_cluster_kmeans(self):
        data = [[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]]
        result = _rule_based_solve("kmeans", data)
        assert result is not None
        assert result.output.get("n_clusters", 0) > 0

    def test_cluster_group(self):
        data = [[1], [2], [100], [101]]
        result = _rule_based_solve("group data", data)
        assert result is not None


@pytest.mark.skipif(not _have_sklearn, reason="sklearn not installed")
class TestClassificationSolver:
    def test_classify_basic(self):
        data = {
            "X_train": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "y_train": [0, 0, 1, 1],
            "X_test": [[0, 0], [1, 1]]
        }
        result = _rule_based_solve("classify items", data)
        assert result is not None
        assert len(result.output) == 2

    def test_classify_knn(self):
        data = {
            "X_train": [[i] for i in range(10)],
            "y_train": [i % 2 for i in range(10)],
            "X_test": [[3], [7]]
        }
        result = _rule_based_solve("knn classification", data)
        assert result is not None

    def test_classify_predict_class(self):
        data = {
            "X_train": [[1], [2], [3], [4]],
            "y_train": ["a", "a", "b", "b"],
            "X_test": [[1.5], [3.5]]
        }
        result = _rule_based_solve("knn predict class", data)
        assert result is not None


@pytest.mark.skipif(not _have_sklearn, reason="sklearn not installed")
class TestRegressionSolver:
    def test_regress_basic(self):
        data = {
            "X_train": [[i] for i in range(10)],
            "y_train": [i * 2.0 for i in range(10)],
            "X_test": [[5], [10]]
        }
        result = _rule_based_solve("linear regression", data)
        assert result is not None
        assert len(result.output) == 2
        assert abs(result.output[0] - 10.0) < 1.0
        assert abs(result.output[1] - 20.0) < 1.0

    def test_regress_forecast(self):
        data = {
            "X_train": [[i] for i in range(5)],
            "y_train": [i * 3.0 for i in range(5)],
            "X_test": [[2], [4]]
        }
        result = _rule_based_solve("forecast values", data)
        assert result is not None

    def test_regress_predict(self):
        data = {
            "X_train": [[i] for i in range(10)],
            "y_train": [i * 1.5 for i in range(10)],
            "X_test": [[6]]
        }
        result = _rule_based_solve("predict value", data)
        assert result is not None
        assert abs(result.output[0] - 9.0) < 1.0


class TestStringMatchingSolver:
    def test_string_match_basic(self):
        data = {"text": "hello world", "pattern": "world"}
        result = _rule_based_solve("string match", data)
        assert result.output == 6
        assert result.algorithm == "naive_string_match"

    def test_string_pattern(self):
        data = {"text": "abcdef", "pattern": "cd"}
        result = _rule_based_solve("find pattern", data)
        assert result.output == 2

    def test_string_substring(self):
        data = {"string": "test string", "substring": "str"}
        result = _rule_based_solve("find substring", data)
        assert result.output == 5

    def test_string_not_found(self):
        data = {"text": "hello", "pattern": "xyz"}
        result = _rule_based_solve("string match", data)
        assert result.output == -1


@pytest.mark.skipif(not _have_scipy, reason="scipy not installed")
class TestImageProcessingSolver:
    def test_gaussian_blur(self):
        data = {"image": [[1, 2], [3, 4]]}
        result = _rule_based_solve("blur image", data)
        assert result is not None
        assert result.algorithm == "gaussian_blur"

    def test_denoise(self):
        data = {"image": [[1, 2], [3, 4]]}
        result = _rule_based_solve("denoise", data)
        assert result is not None

    def test_edge_detection(self):
        data = {"image": [[1, 2], [3, 4]]}
        result = _rule_based_solve("edge detection", data)
        assert result is not None


class TestPrimeSolver:
    def test_prime_number(self):
        data = {"num": 7}
        result = _rule_based_solve("is prime", data)
        assert result.output is True
        assert result.algorithm == "trial_division"

    def test_not_prime(self):
        data = {"num": 8}
        result = _rule_based_solve("is prime", data)
        assert result.output is False

    def test_factor(self):
        data = {"num": 17}
        result = _rule_based_solve("prime factor check", data)
        assert result.output is True

    def test_prime_one(self):
        data = {"num": 1}
        result = _rule_based_solve("is prime", data)
        assert result is None or result.output is False


class TestFallbackSolver:
    def test_fallback_sort_list(self):
        result = _rule_based_solve("something random", [3, 1, 2])
        assert result is None

    def test_fallback_no_data(self):
        result = _rule_based_solve("solve this problem", None)
        assert result is None

    def test_fallback_string_data(self):
        result = _rule_based_solve("analyze", "some text")
        assert result is None

    def test_fallback_dict_data(self):
        result = _rule_based_solve("unknown problem", {"foo": "bar"})
        assert result is None


class TestConfidence:
    def test_solution_has_confidence(self):
        result = _rule_based_solve("sort", [3, 1, 2])
        assert result.confidence == 0.85

    def test_solution_has_iterations(self):
        result = _rule_based_solve("sort", [3, 1, 2])
        assert result.iterations == 1

    def test_solution_is_not_novel(self):
        result = _rule_based_solve("sort", [3, 1, 2])
        assert result.is_novel is False
