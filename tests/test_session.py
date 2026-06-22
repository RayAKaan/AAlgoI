"""Tests for aalgoi._session module."""
import pytest
from aalgoi._session import MindSession
from aalgoi._core import Mind
from aalgoi._result import SolveResult
import tempfile
import os


class TestMindSession:
    def test_session_solve(self, tmp_path):
        with MindSession(tmp_path / "session") as session:
            r = session.solve("sort", [3, 1, 2])
            assert r.ok

    def test_session_learn_with_expected_mismatch(self, tmp_path):
        with MindSession(tmp_path / "session2") as session:
            r = session.learn("sort", [3, 1, 2], expected=[9, 9, 9])
            assert r.error is not None
            assert r.confidence < 0.8

    def test_session_learn_with_expected_match(self, tmp_path):
        with MindSession(tmp_path / "session3") as session:
            r = session.learn("sort", [3, 1, 2], expected=[1, 2, 3])
            assert r.ok
            assert r.error is None

    def test_session_status(self, tmp_path):
        with MindSession(tmp_path / "session4") as session:
            session.solve("sort", [3, 1, 2])
            status = session.status()
            assert "Solved:" in status
