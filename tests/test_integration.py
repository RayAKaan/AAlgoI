"""Integration tests — full pipeline from input to output."""
import pytest
from importlib.util import find_spec
import warnings

import aalgoi
from aalgoi._core import Mind
from aalgoi._session import MindSession
from aalgoi._result import SolveResult

_have_sklearn = find_spec("sklearn") is not None
_have_scipy = find_spec("scipy") is not None


class TestFullSolvePipeline:
    def test_solve_sort_list(self):
        r = aalgoi.solve("sort the array", [3, 1, 4, 1, 5, 9, 2, 6])
        assert r.ok
        assert r.output == [1, 1, 2, 3, 4, 5, 6, 9]
        assert r.algorithm == "tim_sort"
        assert r.complexity == "O(n log n)"
        assert r.confidence > 0

    def test_solve_descending(self):
        r = aalgoi.solve("sort descending", [3, 1, 4, 1, 5])
        assert r.ok
        assert r.output == [5, 4, 3, 1, 1]

    def test_solve_two_sum(self):
        r = aalgoi.solve("two sum", {"nums": [2, 7, 11, 15], "target": 9})
        assert r.ok
        assert r.output == [0, 1]
        assert r.algorithm == "hash_complement"

    def test_solve_kadane(self):
        r = aalgoi.solve("maximum subarray", [-2, 1, -3, 4, -1, 2, 1, -5, 4])
        assert r.ok
        assert r.output == 6
        assert r.algorithm == "kadane"

    def test_solve_gcd(self):
        r = aalgoi.solve("gcd of 48 and 18", {"a": 48, "b": 18})
        assert r.ok
        assert r.output == 6

    def test_solve_lcm(self):
        r = aalgoi.solve("lcm of 4 and 6", {"a": 4, "b": 6})
        assert r.ok
        assert r.output == 12

    def test_solve_path(self):
        graph = {"A": {"B": 1, "C": 4}, "B": {"C": 2}, "C": {}}
        r = aalgoi.solve("shortest path from A to C", {"graph": graph, "start": "A", "end": "C"})
        assert r.ok
        assert r.output[0] == "A"
        assert r.output[-1] == "C"
        assert len(r.output) >= 2

    def test_solve_knapsack(self):
        items = [
            {"name": "a", "weight": 2, "value": 3},
            {"name": "b", "weight": 3, "value": 4},
        ]
        r = aalgoi.solve("knapsack", {"items": items, "capacity": 3})
        assert r.ok

    @pytest.mark.skipif(not _have_sklearn, reason="sklearn not installed")
    def test_solve_clustering(self):
        data = [[1, 1], [1.2, 1.1], [5, 5], [5.1, 5.2]]
        r = aalgoi.solve("cluster data", data)
        assert r.ok
        assert "labels" in r.output
        assert "centers" in r.output

    @pytest.mark.skipif(not _have_sklearn, reason="sklearn not installed")
    def test_solve_classification(self):
        data = {
            "X_train": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "y_train": [0, 0, 1, 1],
            "X_test": [[0, 0], [1, 1]]
        }
        r = aalgoi.solve("classify items", data)
        assert r.ok
        assert len(r.output) == 2

    @pytest.mark.skipif(not _have_sklearn, reason="sklearn not installed")
    def test_solve_regression(self):
        data = {
            "X_train": [[i] for i in range(10)],
            "y_train": [i * 2.0 for i in range(10)],
            "X_test": [[5], [10]]
        }
        r = aalgoi.solve("predict values", data)
        assert r.ok
        assert len(r.output) == 2

    def test_solve_string_match(self):
        data = {"text": "hello world", "pattern": "world"}
        r = aalgoi.solve("string match", data)
        assert r.ok
        assert r.output == 6

    def test_solve_primes(self):
        r = aalgoi.solve("is prime", {"num": 7})
        assert r.ok
        assert r.output is True

    @pytest.mark.skipif(not _have_scipy, reason="scipy not installed")
    def test_solve_image_blur(self):
        import numpy as np
        data = {"image": np.random.rand(10, 10)}
        r = aalgoi.solve("blur image", data)
        assert r.ok
        assert r.algorithm == "gaussian_blur"

    def test_solve_no_data(self):
        r = aalgoi.solve("solve this problem")
        assert not r.ok
        assert "No solution" in (r.error or "")

    def test_solve_empty_list(self):
        r = aalgoi.solve("sort", [])
        assert r.ok
        assert r.output == []


class TestMindIntegration:
    def test_mind_solve_sequence(self, tmp_path):
        m = Mind(tmp_path / "mind")
        r1 = m.solve("sort", [3, 1, 2])
        r2 = m.solve("two sum", {"nums": [2, 7, 11, 15], "target": 9})
        r3 = m.solve("maximum subarray", [-2, 1, -3, 4, -1, 2, 1, -5, 4])
        assert r1.ok
        assert r2.ok
        assert r3.ok
        assert m._solve_count == 3
        assert m._success_count == 3

    def test_mind_solve_then_status(self, tmp_path):
        m = Mind(tmp_path / "mind")
        m.solve("sort", [3, 1, 2])
        m.solve("gcd", {"a": 12, "b": 8})
        status = m.status()
        assert "Solved:      2" in status
        assert "Success rate: 100%" in status

    def test_mind_solve_then_learn(self, tmp_path):
        m = Mind(tmp_path / "mind")
        r = m.solve("sort", [3, 1, 2])
        assert r.ok
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = m.learn("sort", [3, 1, 2])
            assert any("deprecated" in str(x.message).lower() for x in w)

    def test_mind_train_checkpoint_rollback_no_torch(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.train(epochs=1)
        assert result["status"] == "no_training_available"
        assert m.checkpoint() is None
        roll = m.rollback()
        assert roll["success"] is False
        assert "no_mind_loaded" in roll.get("error", "")

    def test_mind_share_receive(self, tmp_path):
        m = Mind(tmp_path / "mind")
        count = m.share()
        assert count == 0
        assert (tmp_path / "mind" / "outbox").exists()
        result = m.receive()
        assert result["updates_processed"] == 0
        assert (tmp_path / "mind" / "inbox").exists()

    def test_mind_algorithms_properties(self, tmp_path):
        m = Mind(tmp_path / "mind")
        algos = m.algorithms
        for name, algo in algos.items():
            assert algo.name == name
            assert algo.time_complexity
            assert algo.space_complexity
            assert isinstance(algo.principles, list)
            assert isinstance(algo.best_for, list)

    def test_mind_reuse(self, tmp_path):
        m = Mind(tmp_path / "mind")
        r1 = m.solve("sort", [3, 1, 2])
        r2 = m.solve("sort", [5, 4, 3, 2, 1])
        r3 = m.solve("sort", [1])
        assert r1.output == [1, 2, 3]
        assert r2.output == [1, 2, 3, 4, 5]
        assert r3.output == [1]


class TestSessionIntegration:
    def test_session_multiple_solves(self, tmp_path):
        with MindSession(tmp_path / "session") as session:
            r1 = session.solve("sort", [3, 1, 2])
            r2 = session.solve("two sum", {"nums": [1, 2, 3], "target": 3})
            assert r1.ok
            assert r2.ok

    def test_session_learn_with_mismatch(self, tmp_path):
        with MindSession(tmp_path / "session") as session:
            r = session.learn("sort", [3, 1, 2], expected=[9, 9, 9])
            assert r.error is not None
            assert r.confidence < 0.8

    def test_session_learn_with_match(self, tmp_path):
        with MindSession(tmp_path / "session") as session:
            r = session.learn("sort", [3, 1, 2], expected=[1, 2, 3])
            assert r.ok
            assert r.error is None

    def test_session_status(self, tmp_path):
        with MindSession(tmp_path / "session") as session:
            session.solve("sort", [3, 1, 2])
            session.solve("two sum", {"nums": [1, 2, 3], "target": 3})
            status = session.status()
            assert "Duration:" in status
            assert "Solved:" in status

    def test_session_exception_handling(self, tmp_path):
        with MindSession(tmp_path / "session") as session:
            session.solve("sort", [3, 1, 2])
            r = session.solve("unknown_problem_xyz")


class TestShortcutsIntegration:
    def test_sort_shortcuts(self):
        from aalgoi import shortcuts
        assert shortcuts.sort([3, 1, 2]) == [1, 2, 3]
        assert shortcuts.sort([3, 1, 2], reverse=True) == [3, 2, 1]
        data = [{"name": "b", "val": 2}, {"name": "a", "val": 1}]
        assert shortcuts.sort_by(data, "val")[0]["val"] == 1
        ranks = shortcuts.rank([30, 10, 20])
        assert ranks == [(1, 10), (2, 20), (3, 30)]

    def test_search_shortcut(self):
        from aalgoi import shortcuts
        assert shortcuts.search([1, 2, 3, 4, 5], 3) == 2
        assert shortcuts.search([1, 2, 3], 99) == -1

    def test_path_shortcuts(self):
        from aalgoi import shortcuts
        graph = {"A": {"B": 1, "C": 4}, "B": {"C": 2}, "C": {}}
        path = shortcuts.path(graph, "A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"
        all_p = shortcuts.all_paths(graph, "A", "C")
        assert len(all_p) >= 1
        dist = shortcuts.distance(graph, "A", "C")
        assert dist is not None
        assert dist >= 1

    def test_knapsack_shortcut(self):
        from aalgoi import shortcuts
        items = [
            {"name": "a", "weight": 2, "value": 3},
            {"name": "b", "weight": 3, "value": 4},
            {"name": "c", "weight": 4, "value": 5},
        ]
        result = shortcuts.knapsack(items, capacity=5)
        assert "selected" in result
        assert "value" in result
        assert "weight" in result
        assert result["weight"] <= 5

    def test_optimization_shortcuts(self):
        from aalgoi import shortcuts
        min_result = shortcuts.minimize(lambda x: (x - 3) ** 2, bounds=(0, 10))
        assert abs(min_result - 3.0) < 0.1
        max_result = shortcuts.maximize(lambda x: -(x - 5) ** 2, bounds=(0, 10))
        assert abs(max_result - 5.0) < 0.5

    def test_ml_shortcuts(self):
        from aalgoi import shortcuts
        data = [[1, 1], [1.2, 1.1], [5, 5], [5.1, 5.2]]
        cluster_result = shortcuts.cluster(data, n=2)
        assert len(cluster_result["labels"]) == 4
        assert len(cluster_result["centers"]) == 2
        X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_train = [0, 0, 1, 1]
        X_test = [[0, 0], [1, 1]]
        class_result = shortcuts.classify(X_train, y_train, X_test)
        assert len(class_result) == 2
        X_train = [[i] for i in range(10)]
        y_train = [i * 2.0 for i in range(10)]
        X_test = [[5], [10]]
        reg_result = shortcuts.regress(X_train, y_train, X_test)
        assert len(reg_result) == 2
        assert abs(reg_result[0] - 10.0) < 2.0
        assert abs(reg_result[1] - 20.0) < 2.0

    def test_compare_shortcut(self):
        from aalgoi import shortcuts
        result = shortcuts.compare("sort", "search", data=[3, 1, 2])
        assert "sort" in result
        assert "time_ms" in result["sort"]
        assert "search" in result
        assert "time_ms" in result["search"]

    def test_compare_unknown_algorithm(self):
        from aalgoi import shortcuts
        result = shortcuts.compare("nonexistent", data=[1, 2, 3])
        assert "nonexistent" in result
        assert "time_ms" in result["nonexistent"]

    def test_why_shortcut(self):
        from aalgoi import shortcuts
        r = aalgoi.solve("sort", [3, 1, 2])
        explanation = shortcuts.why(r)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "tim_sort" in explanation.lower() or "algorithm" in explanation.lower()
