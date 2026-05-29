"""Tests for federation server API."""
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

# Point database to temp dir for test isolation
import server.database as db_mod

@pytest.fixture(autouse=True)
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = Path(tmp) / "test_federation.db"
        yield
        db_mod.DB_PATH = old_path


@pytest.fixture
def client():
    from server.federation_api import app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.4.0"


def test_ingest_metric(client):
    payload = {
        "algorithm": "quicksort",
        "problem_type": "SORTING",
        "success": True,
        "execution_time_ms": 0.5,
        "data_size": 100,
        "user_id_hash": "abc123hash"
    }
    resp = client.post("/metrics/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert "id" in data


def test_ingest_metric_missing_fields(client):
    payload = {"algorithm": "quicksort"}
    resp = client.post("/metrics/ingest", json=payload)
    assert resp.status_code == 422


def test_aggregate_empty(client):
    resp = client.get("/metrics/aggregate/nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["algorithm"] == "nonexistent"
    assert data["samples"] == 0


def test_aggregate_with_data(client):
    payload = {
        "algorithm": "timsort",
        "problem_type": "SORTING",
        "success": True,
        "execution_time_ms": 0.3,
        "data_size": 100,
        "user_id_hash": "user1"
    }
    client.post("/metrics/ingest", json=payload)

    payload2 = payload.copy()
    payload2["success"] = False
    payload2["execution_time_ms"] = 1.0
    payload2["user_id_hash"] = "user2"
    client.post("/metrics/ingest", json=payload2)

    resp = client.get("/metrics/aggregate/timsort")
    assert resp.status_code == 200
    data = resp.json()
    assert data["samples"] == 2
    assert data["avg_success"] == 0.5
    assert data["avg_time_ms"] == 0.65


def test_sync_state(client):
    resp = client.get("/sync/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "global_version" in data
    assert "new_algorithms_last_hour" in data
    assert "timestamp" in data


def test_submit_algorithm(client):
    code = "def process(data): return sorted(data)"
    payload = {
        "name": "synth_test_v1",
        "code": code,
        "metadata": {"problem_types": ["SORTING"], "complexity": "O(n log n)"},
        "embedding": [0.1, 0.2, 0.3],
        "submitted_by": "testuser",
        "code_hash": "testhash1",
    }
    resp = client.post("/algorithms/submit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"


def test_submit_duplicate(client):
    payload = {
        "name": "synth_dup",
        "code": "def process(data): return sorted(data)",
        "metadata": {"problem_types": ["SORTING"]},
        "embedding": [0.1],
        "submitted_by": "user1",
        "code_hash": "duphash1",
    }
    client.post("/algorithms/submit", json=payload)
    resp = client.post("/algorithms/submit", json=payload)
    assert resp.status_code == 409


def test_get_status_found(client):
    payload = {
        "name": "synth_status_test",
        "code": "def process(d): return d",
        "metadata": {"problem_types": ["SORTING"]},
        "embedding": [0.5],
        "submitted_by": "test",
        "code_hash": "statushash1",
    }
    client.post("/algorithms/submit", json=payload)

    resp = client.get("/algorithms/status/statushash1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("pending", "passed")


def test_get_status_not_found(client):
    resp = client.get("/algorithms/status/nonexistent")
    assert resp.status_code == 404


def test_security_headers(client):
    resp = client.get("/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("x-xss-protection") == "1; mode=block"


def test_multiple_metrics_ingestion(client):
    for i in range(5):
        client.post("/metrics/ingest", json={
            "algorithm": "merge_sort",
            "problem_type": "SORTING",
            "success": True,
            "execution_time_ms": 0.1 * i,
            "data_size": 100,
            "user_id_hash": f"user{i}",
        })

    resp = client.get("/metrics/aggregate/merge_sort")
    assert resp.status_code == 200
    assert resp.json()["samples"] == 5


def test_server_exposes_endpoints(client):
    """OpenAPI spec should list all endpoints."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json().get("paths", {})
    assert "/health" in paths
    assert "/metrics/ingest" in paths
    assert "/metrics/aggregate/{algorithm}" in paths
    assert "/algorithms/submit" in paths
    assert "/algorithms/status/{code_hash}" in paths
    assert "/sync/state" in paths


def test_invalid_method(client):
    """Invalid HTTP method returns 405."""
    resp = client.put("/health")
    assert resp.status_code == 405
