"""Performance benchmark tests — catch regressions in solve speed."""
import pytest
import time

import aalgoi
from aalgoi._core import Mind
from aalgoi._data import normalize


class TestSolvePerformance:
    _warmed_up = False

    def _warmup(self):
        if not TestSolvePerformance._warmed_up:
            aalgoi.solve("sort", [3, 1, 2])
            TestSolvePerformance._warmed_up = True

    def test_solve_sort_small(self):
        self._warmup()
        data = list(range(100, 0, -1))
        start = time.perf_counter()
        aalgoi.solve("sort", data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5

    def test_solve_sort_medium(self):
        self._warmup()
        data = list(range(1000, 0, -1))
        start = time.perf_counter()
        aalgoi.solve("sort", data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5

    def test_solve_two_sum(self):
        self._warmup()
        data = {"nums": list(range(1000)), "target": 1500}
        start = time.perf_counter()
        aalgoi.solve("two sum", data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5

    def test_solve_kadane(self):
        self._warmup()
        data = list(range(-500, 500))
        start = time.perf_counter()
        aalgoi.solve("maximum subarray", data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5

    def test_solve_gcd(self):
        self._warmup()
        data = {"a": 123456789, "b": 987654321}
        start = time.perf_counter()
        aalgoi.solve("gcd", data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5

    def test_solve_path(self):
        self._warmup()
        graph = {str(i): {str(j): 1 for j in range(max(0, i-2), min(20, i+3)) if j != i} for i in range(20)}
        data = {"graph": graph, "start": "0", "end": "19"}
        start = time.perf_counter()
        aalgoi.solve("shortest path", data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5


class TestNormalizePerformance:
    def test_normalize_large_list(self):
        data = list(range(10000))
        start = time.perf_counter()
        normalize(data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1

    def test_normalize_deeply_nested(self):
        data = {"a": {"b": {"c": {"d": [1, 2, 3, 4, 5]}}}}
        start = time.perf_counter()
        normalize(data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.01

    def test_normalize_numpy_array(self):
        import numpy as np
        data = np.random.rand(100, 100)
        start = time.perf_counter()
        normalize(data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05


class TestMindPerformance:
    def test_mind_instantiation(self):
        tmp = __import__('tempfile').mkdtemp()
        start = time.perf_counter()
        m = Mind(tmp)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1

    def test_mind_solve_multiple(self):
        tmp = __import__('tempfile').mkdtemp()
        m = Mind(tmp)
        start = time.perf_counter()
        for i in range(10):
            m.solve("sort", list(range(100, 0, -1)))
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5

    def test_mind_status(self):
        tmp = __import__('tempfile').mkdtemp()
        m = Mind(tmp)
        start = time.perf_counter()
        m.status()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05


class TestShortcutsPerformance:
    def test_shortcuts_sort(self):
        from aalgoi import shortcuts
        data = list(range(1000, 0, -1))
        start = time.perf_counter()
        shortcuts.sort(data)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.01

    def test_shortcuts_search(self):
        from aalgoi import shortcuts
        data = list(range(10000))
        start = time.perf_counter()
        shortcuts.search(data, 9999)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.01

    def test_shortcuts_minimize(self):
        from aalgoi import shortcuts
        start = time.perf_counter()
        shortcuts.minimize(lambda x: x ** 2, bounds=(-10, 10), steps=1000)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1
