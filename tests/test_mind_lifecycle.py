"""Comprehensive tests for Mind lifecycle: train, checkpoint, rollback, share, receive."""
import pytest
import warnings
import json

from aalgoi._core import Mind, BenchmarkReport


class TestMindLifecycle:
    def test_mind_creates_directory(self, tmp_path):
        m = Mind(tmp_path / "mind")
        assert (tmp_path / "mind").exists()

    def test_mind_default_path(self):
        m = Mind()
        assert m.path.exists()

    def test_mind_status(self, tmp_path):
        m = Mind(tmp_path / "mind")
        status = m.status()
        assert "Algorithms:" in status
        assert "Principles:" in status
        assert "Problems:" in status
        assert "Solved:" in status
        assert "Success rate:" in status

    def test_mind_status_after_solve(self, tmp_path):
        m = Mind(tmp_path / "mind")
        m.solve("sort", [3, 1, 2])
        status = m.status()
        assert "Solved:      1" in status
        assert "Success rate: 100%" in status

    def test_mind_algorithms(self, tmp_path):
        m = Mind(tmp_path / "mind")
        algos = m.algorithms
        assert isinstance(algos, dict)
        assert len(algos) > 0
        assert "tim_sort" in algos

    def test_mind_principles(self, tmp_path):
        m = Mind(tmp_path / "mind")
        principles = m.principles
        assert isinstance(principles, list)
        assert "divide_conquer" in principles
        assert "dynamic_programming" in principles
        assert principles == sorted(principles)

    def test_mind_problems(self, tmp_path):
        m = Mind(tmp_path / "mind")
        problems = m.problems
        assert isinstance(problems, list)
        assert "SORTING" in problems
        assert "PATHFINDING" in problems
        assert problems == sorted(problems)


class TestMindLearnDeprecation:
    def test_learn_emits_deprecation_warning(self, tmp_path):
        m = Mind(tmp_path / "mind")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.learn("sort", [3, 1, 2])
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_learn_still_works(self, tmp_path):
        m = Mind(tmp_path / "mind")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = m.learn("sort", [3, 1, 2])
            assert result.ok
            assert result.output == [1, 2, 3]


class TestMindVerify:
    def test_solve_and_check_output(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.solve("sort", [3, 1, 2])
        assert result.ok
        assert result.output == [1, 2, 3]

    def test_solve_mismatch(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.solve("sort", [3, 1, 2])
        assert result.output != [9, 9, 9]

    def test_solve_returns_without_expected(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.solve("sort", [3, 1, 2])
        assert result.ok


class TestMindTrainNoTorch:
    def test_train_returns_no_training_available(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.train(epochs=10)
        assert result["status"] == "no_training_available"
        assert result["epochs"] == 0

    def test_train_with_kwargs(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.train(epochs=5, augmentations=3)
        assert result["status"] == "no_training_available"


class TestMindCheckpointNoTorch:
    def test_checkpoint_returns_none(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.checkpoint()
        assert result is None

    def test_checkpoint_with_name(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.checkpoint("my_checkpoint")
        assert result is None


class TestMindRollbackNoTorch:
    def test_rollback_returns_error(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.rollback("last_good")
        assert result["success"] is False
        assert "no_mind_loaded" in result["error"]

    def test_rollback_custom_target(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.rollback("custom_target")
        assert result["success"] is False


class TestMindShare:
    def test_share_empty_outbox(self, tmp_path):
        m = Mind(tmp_path / "mind")
        count = m.share()
        assert count == 0
        assert (tmp_path / "mind" / "outbox").exists()

    def test_share_with_files(self, tmp_path):
        m = Mind(tmp_path / "mind")
        count = m.share()
        assert count == 0
        outbox = tmp_path / "mind" / "outbox"
        (outbox / "update1.json").write_text("{}")
        (outbox / "update2.json").write_text("{}")
        count2 = m.share()
        assert count2 == 2


class TestMindReceive:
    def test_receive_empty_inbox(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.receive()
        assert result["updates_processed"] == 0
        assert result["algorithms_imported"] == 0
        assert (tmp_path / "mind" / "inbox").exists()

    def test_receive_with_files(self, tmp_path):
        m = Mind(tmp_path / "mind")
        result = m.receive()
        assert result["updates_processed"] == 0
        inbox = tmp_path / "mind" / "inbox"
        (inbox / "update1.json").write_text("{}")
        (inbox / "update2.json").write_text("{}")
        (inbox / "update3.json").write_text("{}")
        result2 = m.receive()
        assert result2["updates_processed"] == 3


class TestMindMultipleSolves:
    def test_solve_count_increments(self, tmp_path):
        m = Mind(tmp_path / "mind")
        m.solve("sort", [3, 1, 2])
        m.solve("two sum", {"nums": [1, 2, 3], "target": 3})
        status = m.status()
        assert "Solved:      2" in status

    def test_success_rate(self, tmp_path):
        m = Mind(tmp_path / "mind")
        m.solve("sort", [3, 1, 2])
        m.solve("sort", [1, 2, 3])
        m.solve("unknown_problem_xyz")
        status = m.status()
        assert "Success rate:" in status


class TestAlgorithmInfo:
    def test_algorithm_info_repr(self, tmp_path):
        m = Mind(tmp_path / "mind")
        algo = m.algorithms["tim_sort"]
        repr_str = repr(algo)
        assert "tim_sort" in repr_str
        assert "O(n log n)" in repr_str

    def test_algorithm_info_display(self, tmp_path):
        m = Mind(tmp_path / "mind")
        algo = m.algorithms["tim_sort"]
        display = algo.display()
        assert "tim_sort" in display
        assert "O(n log n)" in display
        assert "O(n)" in display
        assert "divide_conquer" in display


class TestBenchmarkReport:
    def test_benchmark_report_repr(self):
        report = BenchmarkReport({"total": 10, "correct": 8, "accuracy": 0.8})
        repr_str = repr(report)
        assert "10" in repr_str
        assert "8" in repr_str
        assert "80%" in repr_str

    def test_benchmark_report_details(self):
        report = BenchmarkReport({
            "total": 5,
            "correct": 3,
            "failed": 1,
            "errors": 1,
            "accuracy": 0.6,
            "by_domain": {"SORTING": 2, "SEARCHING": 1},
            "problems": ["p1", "p2", "p3", "p4", "p5"]
        })
        assert report.total == 5
        assert report.correct == 3
        assert report.failed == 1
        assert report.errors == 1
        assert report.accuracy == 0.6
        assert report.by_domain == {"SORTING": 2, "SEARCHING": 1}
        assert report.problems == ["p1", "p2", "p3", "p4", "p5"]

    def test_benchmark_report_defaults(self):
        report = BenchmarkReport({})
        assert report.total == 0
        assert report.correct == 0
        assert report.failed == 0
        assert report.errors == 0
        assert report.accuracy == 0.0
        assert report.by_domain == {}
        assert report.problems == []
