"""Tests for aalgoi.shortcuts module."""
import pytest
from aalgoi import shortcuts


class TestSort:
    def test_sort_basic(self):
        assert shortcuts.sort([3, 1, 2]) == [1, 2, 3]

    def test_sort_reverse(self):
        assert shortcuts.sort([3, 1, 2], reverse=True) == [3, 2, 1]

    def test_sort_by_key(self):
        data = [{"name": "b", "val": 2}, {"name": "a", "val": 1}]
        result = shortcuts.sort_by(data, "val")
        assert result[0]["val"] == 1

    def test_rank(self):
        result = shortcuts.rank([30, 10, 20])
        assert result == [(1, 10), (2, 20), (3, 30)]


class TestSearch:
    def test_search_found(self):
        assert shortcuts.search([1, 2, 3, 4, 5], 3) == 2

    def test_search_not_found(self):
        assert shortcuts.search([1, 2, 3], 99) == -1


class TestPath:
    def test_path_basic(self):
        graph = {"A": {"B": 1, "C": 4}, "B": {"C": 2}, "C": {}}
        result = shortcuts.path(graph, "A", "C")
        assert result is not None
        assert result[0] == "A"
        assert result[-1] == "C"

    def test_all_paths(self):
        graph = {"A": ["B", "C"], "B": ["C"], "C": []}
        result = shortcuts.all_paths(graph, "A", "C")
        assert len(result) >= 1


class TestOptimization:
    def test_minimize(self):
        result = shortcuts.minimize(lambda x: (x - 3) ** 2, bounds=(0, 10))
        assert abs(result - 3.0) < 0.1

    def test_maximize(self):
        result = shortcuts.maximize(lambda x: -(x - 5) ** 2, bounds=(0, 10))
        assert abs(result - 5.0) < 0.5

    def test_knapsack(self):
        items = [
            {"name": "a", "weight": 2, "value": 3},
            {"name": "b", "weight": 3, "value": 4},
            {"name": "c", "weight": 4, "value": 5},
        ]
        result = shortcuts.knapsack(items, capacity=5)
        assert "selected" in result
        assert "value" in result
        assert "weight" in result


class TestML:
    def test_cluster(self):
        data = [[1, 1], [1.2, 1.1], [5, 5], [5.1, 5.2]]
        result = shortcuts.cluster(data, n=2)
        assert "labels" in result
        assert len(result["labels"]) == 4

    def test_classify(self):
        X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_train = [0, 0, 1, 1]
        X_test = [[0, 0], [1, 1]]
        result = shortcuts.classify(X_train, y_train, X_test)
        assert len(result) == 2

    def test_regress(self):
        X_train = [[i] for i in range(10)]
        y_train = [i * 2.0 for i in range(10)]
        X_test = [[5], [10]]
        result = shortcuts.regress(X_train, y_train, X_test)
        assert len(result) == 2


class TestCompare:
    def test_compare_sort(self):
        result = shortcuts.compare("sort", data=[3, 1, 2])
        assert "sort" in result
        assert "time_ms" in result["sort"]

    def test_compare_unknown_algorithm(self):
        result = shortcuts.compare("nonexistent_algo", data=[1, 2, 3])
        assert "nonexistent_algo" in result
