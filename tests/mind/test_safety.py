import pytest
from pathlib import Path
from unittest.mock import MagicMock

from aalgoi.core.mind.safety_manager import MindSafetyManager


@pytest.fixture
def manager(tmp_path):
    return MindSafetyManager(tmp_path)


class TestMindSafetyManagerCreation:
    def test_creates_with_user_dir(self, tmp_path):
        m = MindSafetyManager(tmp_path)
        assert m.user_dir == tmp_path

    def test_default_solve_count_zero(self, manager):
        assert manager.solve_count == 0

    def test_default_history_empty(self, manager):
        assert manager.history == []


class TestCheckpointInterval:
    def test_no_checkpoint_before_interval(self, manager):
        mind = MagicMock()
        mind.last_solve_time = 1.0
        ckpt_dir = manager.user_dir / "checkpoints"
        for i in range(49):
            manager.auto_checkpoint(mind, i + 1)
        assert not ckpt_dir.is_dir() or len(list(ckpt_dir.glob("*.json"))) == 0

    def test_checkpoint_at_interval(self, manager):
        mind = MagicMock()
        mind.last_solve_time = 1.0
        for i in range(50):
            manager.auto_checkpoint(mind, i + 1)
        ckpt_dir = manager.user_dir / "checkpoints"
        assert len(list(ckpt_dir.glob("*.json"))) == 1

    def test_checkpoint_multiple_intervals(self, manager):
        mind = MagicMock()
        mind.last_solve_time = 1.0
        for i in range(100):
            manager.auto_checkpoint(mind, i + 1)
        ckpt_dir = manager.user_dir / "checkpoints"
        assert len(list(ckpt_dir.glob("*.json"))) == 2

    def test_checkpoint_at_correct_counts(self, manager):
        mind = MagicMock()
        mind.last_solve_time = 1.0
        for i in range(100):
            manager.auto_checkpoint(mind, i + 1)
        ckpt_dir = manager.user_dir / "checkpoints"
        stems = {p.stem for p in ckpt_dir.glob("*.json")}
        assert "checkpoint_50" in stems
        assert "checkpoint_100" in stems


class TestDegradationDetection:
    def test_not_enough_history_returns_false(self, manager):
        mind = MagicMock()
        mind.last_solve_time = 1.0
        for i in range(29):
            manager.auto_checkpoint(mind, i + 1)
        assert manager.detect_quality_degradation() is False

    def test_warning_at_2_sigma(self, manager):
        # Push 29 fast solves then 1 slow solve
        for i in range(29):
            manager._history.append({"solve_count": i + 1, "time": 1.0})
        manager._history.append({"solve_count": 30, "time": 10.0})
        # mean=1.29, std~1.64, 2*std=3.28 -> threshold=4.57, 10.0 > 4.57 -> warning
        assert manager.detect_quality_degradation() is True

    def test_auto_rollback_at_3_sigma(self, manager):
        for i in range(29):
            manager._history.append({"solve_count": i + 1, "time": 1.0})
        manager._history.append({"solve_count": 30, "time": 100.0})
        ckpt_dir = manager.user_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(ckpt_dir / "checkpoint_25.json", "w") as f:
            json.dump({"solve_count": 25}, f)
        assert manager.detect_quality_degradation() is True

    def test_no_degradation_normal(self, manager):
        for i in range(30):
            manager._history.append({"solve_count": i + 1, "time": 1.0})
        assert manager.detect_quality_degradation() is False


class TestRollback:
    def test_rollback_no_checkpoint_dir(self, manager):
        with pytest.raises(ValueError, match="No checkpoint directory found"):
            manager.rollback()

    def test_rollback_no_checkpoint_files(self, manager):
        ckpt_dir = manager.user_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="No checkpoints available"):
            manager.rollback()

    def test_rollback_unknown_target(self, manager):
        with pytest.raises(ValueError, match="Unknown rollback target"):
            manager.rollback("specific")

    def test_rollback_loads_latest_checkpoint(self, manager):
        ckpt_dir = manager.user_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(ckpt_dir / "checkpoint_50.json", "w") as f:
            json.dump({"solve_count": 50}, f)
        with open(ckpt_dir / "checkpoint_100.json", "w") as f:
            json.dump({"solve_count": 100}, f)
        # Should not raise
        manager.rollback()
