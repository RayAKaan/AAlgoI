"""Unit tests for registry sync, token manager, and config loading."""
import hashlib
import os
from unittest.mock import patch, MagicMock, ANY

import pytest

from core.registry_sync import GitHubRegistrySync, _load_registry_config
from core.token_manager import TokenManager


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    from core.registry_manager import DynamicRegistry
    return DynamicRegistry(None)


@pytest.fixture
def embedder():
    from core.algorithm_embedder import AlgorithmEmbedder
    return AlgorithmEmbedder()


@pytest.fixture
def sync(registry, embedder):
    s = GitHubRegistrySync(registry, embedder, config={})
    return s


# ── TokenManager ─────────────────────────────────────────────────────────────


class TestTokenManager:
    def teardown_method(self):
        TokenManager.clear_cache()
        try:
            del os.environ["GITHUB_TOKEN"]
        except KeyError:
            pass

    def test_get_token_returns_none_when_unset(self):
        assert TokenManager.get_token() is None

    def test_get_token_returns_value_when_set(self):
        os.environ["GITHUB_TOKEN"] = "ghp_test123"
        assert TokenManager.get_token() == "ghp_test123"

    def test_require_token_raises_when_unset(self):
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            TokenManager.require_token()

    def test_require_token_returns_value_when_set(self):
        os.environ["GITHUB_TOKEN"] = "ghp_test456"
        assert TokenManager.require_token() == "ghp_test456"

    def test_clear_cache_forces_reread(self):
        os.environ["GITHUB_TOKEN"] = "ghp_first"
        assert TokenManager.get_token() == "ghp_first"
        os.environ["GITHUB_TOKEN"] = "ghp_second"
        assert TokenManager.get_token() == "ghp_first"  # cached
        TokenManager.clear_cache()
        assert TokenManager.get_token() == "ghp_second"  # re-read


# ── Config loading ────────────────────────────────────────────────────────────


class TestRegistryConfig:
    def test_load_registry_config_returns_defaults_on_missing_file(self, tmp_path):
        import json
        from pathlib import Path

        cfg = _load_registry_config()
        # Should return empty dict when no config file found
        assert isinstance(cfg, dict)


# ── Pull ──────────────────────────────────────────────────────────────────────


class TestPull:
    def test_up_to_date(self, sync):
        sync._save_state({"index_version": 5, "algorithm_names": []})
        with patch.object(sync, "_fetch_index", return_value={"version": 5, "algorithms": []}):
            result = sync._pull()
        assert result["status"] == "up_to_date"
        assert result["version"] == 5
        assert sync._backoff_delay == 0

    def test_offline_sets_backoff(self, sync):
        sync._backoff_delay = 0
        with patch.object(sync, "_fetch_index", side_effect=ConnectionError("no route to host")):
            result = sync._pull()
        assert result["status"] == "offline"
        assert sync._backoff_delay > 0

    def test_offline_doubles_backoff(self, sync):
        sync._backoff_delay = 60
        with patch.object(sync, "_fetch_index", side_effect=ConnectionError("still down")):
            sync._pull()
        assert sync._backoff_delay == 120

    def test_offline_caps_backoff(self, sync):
        sync._backoff_delay = sync.backoff_max
        with patch.object(sync, "_fetch_index", side_effect=ConnectionError("still down")):
            sync._pull()
        assert sync._backoff_delay == sync.backoff_max

    def test_downloads_new_algorithms(self, sync):
        real_checksum = "sha256:" + hashlib.sha256(b"print('hello')").hexdigest()
        mock_index = {
            "version": 2,
            "algorithms": [{
                "name": "HelloAlgo",
                "path": "algorithms/test/hello",
                "checksum": real_checksum,
            }],
        }
        mock_metadata = {
            "name": "HelloAlgo",
            "problem_types": ["SORTING"],
        }
        sync._save_state({"index_version": 1, "algorithm_names": []})

        with patch.object(sync, "_fetch_index", return_value=mock_index), \
             patch("requests.get") as mock_get:
            def side_effect(url, **kwargs):
                resp = MagicMock()
                if url.endswith(".py"):
                    resp.text = "print('hello')"
                    resp.raise_for_status = lambda: None
                elif url.endswith(".json"):
                    resp.json.return_value = mock_metadata
                    resp.raise_for_status = lambda: None
                else:
                    resp.raise_for_status.side_effect = ValueError("unexpected url")
                return resp
            mock_get.side_effect = side_effect
            result = sync._pull()

        assert result["status"] == "updated"
        assert result["new_algorithms"] == 1
        assert result["failed"] == 0

    def test_rejects_checksum_mismatch(self, sync):
        wrong_checksum = "sha256:" + hashlib.sha256(b"real code").hexdigest()
        mock_index = {
            "version": 2,
            "algorithms": [{
                "name": "Tampered",
                "path": "algorithms/test/tampered",
                "checksum": wrong_checksum,
            }],
        }
        sync._save_state({"index_version": 1, "algorithm_names": []})

        with patch.object(sync, "_fetch_index", return_value=mock_index), \
             patch("requests.get") as mock_get:
            resp = MagicMock()
            resp.text = "tampered code"
            resp.raise_for_status = lambda: None
            mock_get.return_value = resp
            result = sync._pull()

        assert result["status"] == "updated"
        assert result["new_algorithms"] == 0
        assert result["failed"] == 1
        assert result["details"][0]["reason"] == "checksum_mismatch"

    def test_success_resets_backoff(self, sync):
        sync._backoff_delay = 300
        with patch.object(sync, "_fetch_index", return_value={"version": 0, "algorithms": []}):
            sync._pull()
        assert sync._backoff_delay == 0


# ── Push ──────────────────────────────────────────────────────────────────────


class TestPush:
    def test_rejects_failed_validation(self, sync):
        algo = MagicMock()
        vr = MagicMock()
        vr.passed = False
        result = sync.push_discovered(algo, vr)
        assert result["status"] == "rejected"

    def test_queues_when_no_token(self, sync, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        TokenManager.clear_cache()

        algo = MagicMock()
        algo.name = "TestAlgo"
        vr = MagicMock()
        vr.passed = True

        result = sync.push_discovered(algo, vr)
        assert result["status"] == "queued"
        assert sync._push_queue

    def test_queues_when_offline_backoff(self, sync, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        TokenManager.clear_cache()
        sync._backoff_delay = 300

        algo = MagicMock()
        algo.name = "TestAlgo"
        vr = MagicMock()
        vr.passed = True

        result = sync.push_discovered(algo, vr)
        assert result["status"] == "queued"
        assert sync._push_queue

    def test_submits_with_token(self, sync, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        TokenManager.clear_cache()
        sync._backoff_delay = 0

        algo = MagicMock()
        algo.name = "QuickSortPlus"
        algo.serialize.return_value = "class QuickSortPlus: pass"
        algo.problem_type = "SORTING"

        vr = MagicMock()
        vr.passed = True
        vr.improvement_pct = 15.0
        vr.pass_rate = 0.95
        vr.baseline_name = "quicksort"
        vr.results = {
            "correctness": 1.0,
            "edge_cases": {"all_passed": True},
            "performance": {"beats_baseline": True},
        }

        with patch.object(sync, "_do_push", return_value={"status": "submitted", "pr_url": "https://github.com/pr/1"}):
            result = sync.push_discovered(algo, vr)

        assert result["status"] == "submitted"
        assert "pr_url" in result


# ── Queue flushing ────────────────────────────────────────────────────────────


class TestQueueFlush:
    def test_flush_queued_pushes_when_token_available(self, sync, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_flush")
        TokenManager.clear_cache()
        sync._backoff_delay = 0

        # Add queued entries
        algo = MagicMock()
        algo.name = "QueuedAlgo"
        algo.serialize.return_value = "class QueuedAlgo: pass"
        algo.problem_type = "SORTING"
        vr = MagicMock()
        vr.passed = True
        vr.improvement_pct = 5.0
        vr.pass_rate = 1.0
        vr.baseline_name = "built-in"
        vr.results = {
            "correctness": 1.0,
            "edge_cases": {"all_passed": True},
            "performance": {"beats_baseline": True},
        }

        sync._push_queue.append({"algorithm": algo, "validation_result": vr})

        with patch.object(sync, "_do_push", return_value={"status": "submitted", "pr_url": "https://github.com/pr/2"}):
            sync._flush_queue()

        assert len(sync._push_queue) == 0

    def test_flush_skips_when_no_token(self, sync, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        TokenManager.clear_cache()

        sync._push_queue.append({"algorithm": None, "validation_result": None})
        sync._flush_queue()
        assert len(sync._push_queue) == 1  # not flushed


# ── Status ────────────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_returns_state(self, sync):
        st = sync.status()
        assert "last_sync" in st
        assert "index_version" in st
        assert "push_queue_depth" in st
        assert "backoff_seconds" in st

    def test_status_reflects_queue(self, sync):
        sync._push_queue.append({"test": True})
        assert sync.status()["push_queue_depth"] == 1

    def test_status_reflects_backoff(self, sync):
        sync._backoff_delay = 120
        assert sync.status()["backoff_seconds"] == 120
