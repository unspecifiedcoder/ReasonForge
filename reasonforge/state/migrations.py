"""
ReasonForge - Database Migrations

Simple versioned migration system for the state database.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import List, Tuple

logger = logging.getLogger("reasonforge.state.migrations")

# Each migration is (version, description, SQL)
MIGRATIONS: List[Tuple[int, str, str]] = [
    (
        1,
        "Initial schema — miner_epochs, task_results, submissions, checkpoints, api_keys",
        """
        CREATE TABLE IF NOT EXISTS miner_epochs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch_id INTEGER NOT NULL,
            miner_uid INTEGER NOT NULL,
            s_epoch REAL NOT NULL DEFAULT 0.0,
            peb REAL NOT NULL DEFAULT 0.0,
            rank INTEGER NOT NULL DEFAULT 0,
            streak INTEGER NOT NULL DEFAULT 0,
            tao_earned REAL NOT NULL DEFAULT 0.0,
            created_at REAL NOT NULL DEFAULT 0,
            UNIQUE(epoch_id, miner_uid)
        );

        CREATE TABLE IF NOT EXISTS task_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            epoch_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            difficulty INTEGER NOT NULL DEFAULT 5,
            is_trap INTEGER NOT NULL DEFAULT 0,
            avg_cms REAL NOT NULL DEFAULT 0.0,
            best_miner_uid INTEGER,
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL,
            miner_uid INTEGER NOT NULL,
            cms REAL NOT NULL DEFAULT 0.0,
            quality REAL NOT NULL DEFAULT 0.0,
            accuracy REAL NOT NULL DEFAULT 0.0,
            novelty REAL NOT NULL DEFAULT 0.0,
            efficiency REAL NOT NULL DEFAULT 0.0,
            submission_hash TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch_id INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_id TEXT NOT NULL,
            api_key TEXT NOT NULL UNIQUE,
            owner TEXT NOT NULL DEFAULT '',
            tier TEXT NOT NULL DEFAULT 'free',
            request_limit INTEGER NOT NULL DEFAULT 100,
            requests_used INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL DEFAULT 0
        );
        """,
    ),
    (
        2,
        "Add indexes for performance",
        """
        CREATE INDEX IF NOT EXISTS idx_miner_epochs_uid ON miner_epochs(miner_uid);
        CREATE INDEX IF NOT EXISTS idx_miner_epochs_epoch ON miner_epochs(epoch_id);
        CREATE INDEX IF NOT EXISTS idx_task_results_epoch ON task_results(epoch_id);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_epoch ON checkpoints(epoch_id);
        CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(api_key);
        """,
    ),
]


class MigrationManager:
    """Simple versioned migration manager for SQLite."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_version_table()

    def _ensure_version_table(self) -> None:
        """Create the schema_version table if it doesn't exist."""
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL DEFAULT 0
            )"""
        )
        # Ensure there is exactly one row
        row = self.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            self.conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        self.conn.commit()

    def get_current_version(self) -> int:
        """Get the current schema version."""
        row = self.conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        return row[0] if row else 0

    def apply_pending(self) -> int:
        """Apply all pending migrations. Returns number of migrations applied."""
        current = self.get_current_version()
        applied = 0

        for version, description, sql in MIGRATIONS:
            if version > current:
                logger.info("Applying migration %d: %s", version, description)
                self.conn.executescript(sql)
                self.conn.execute("UPDATE schema_version SET version = ?", (version,))
                self.conn.commit()
                applied += 1
                current = version

        return applied
