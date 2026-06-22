import sqlite3
from pathlib import Path

DB_PATH = Path("federation.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            problem_type TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            execution_time_ms REAL NOT NULL,
            data_size INTEGER,
            user_id_hash TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS algorithm_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding TEXT NOT NULL,
            submitted_by TEXT NOT NULL,
            code_hash TEXT NOT NULL UNIQUE,
            status TEXT DEFAULT 'pending',
            validation_result TEXT,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            promoted_at DATETIME
        );

        CREATE TABLE IF NOT EXISTS algorithm_index (
            name TEXT PRIMARY KEY,
            version TEXT,
            status TEXT,
            problem_types TEXT,
            code_hash TEXT,
            path TEXT,
            last_updated DATETIME
        );

        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.execute(
        "INSERT OR IGNORE INTO sync_state (key, value) VALUES (?, ?)",
        ("last_global_version", "0")
    )
    conn.commit()
    return conn
