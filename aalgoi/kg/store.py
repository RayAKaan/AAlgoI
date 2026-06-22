from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".aalgoi" / "kg" / "store.db"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._init_db()
        return self._conn

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                task TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                success INTEGER NOT NULL,
                validated INTEGER NOT NULL,
                time_ms REAL NOT NULL,
                error TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                failure_reason TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                algorithm TEXT NOT NULL,
                task TEXT NOT NULL,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                total_time_ms REAL DEFAULT 0,
                run_count INTEGER DEFAULT 0,
                avg_time_ms REAL DEFAULT 0,
                PRIMARY KEY (algorithm, task)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self._conn.commit()

    def record_run(self, problem_id: str, task: str, algorithm: str, success: bool, validated: bool, time_ms: float, error: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO runs (problem_id, task, algorithm, success, validated, time_ms, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (problem_id, task, algorithm, int(success), int(validated), time_ms, error, time.time()),
        )
        perf = self.conn.execute(
            "SELECT successes, failures, total_time_ms, run_count FROM performance WHERE algorithm = ? AND task = ?",
            (algorithm, task),
        ).fetchone()
        if perf:
            s, f, tt, rc = perf
            self.conn.execute(
                "UPDATE performance SET successes = ?, failures = ?, total_time_ms = ?, run_count = ?, avg_time_ms = ? WHERE algorithm = ? AND task = ?",
                (s + (1 if success else 0), f + (0 if success else 1), tt + time_ms, rc + 1, (tt + time_ms) / (rc + 1), algorithm, task),
            )
        else:
            self.conn.execute(
                "INSERT INTO performance (algorithm, task, successes, failures, total_time_ms, run_count, avg_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (algorithm, task, 1 if success else 0, 0 if success else 1, time_ms, 1, time_ms),
            )
        self.conn.commit()

    def record_failure(self, problem_id: str, algorithm: str, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO failures (problem_id, algorithm, failure_reason, created_at) VALUES (?, ?, ?, ?)",
            (problem_id, algorithm, reason, time.time()),
        )
        self.conn.commit()

    def get_performance(self, algorithm: str, task: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM performance WHERE algorithm = ? AND task = ?",
            (algorithm, task),
        ).fetchone()
        if row is None:
            return {"successes": 0, "failures": 0, "avg_time_ms": 0.0, "run_count": 0}
        return {
            "successes": row[2],
            "failures": row[3],
            "total_time_ms": row[4],
            "run_count": row[5],
            "avg_time_ms": row[6],
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
