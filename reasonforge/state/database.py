"""
ReasonForge - State Database

SQLite-backed persistent storage for miner epochs, task results,
submissions, checkpoints, and API keys.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("reasonforge.state.database")


class StateDatabase:
    """SQLite state persistence for validators and miners."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS miner_epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch_id INTEGER NOT NULL,
                miner_uid INTEGER NOT NULL,
                s_epoch REAL NOT NULL DEFAULT 0.0,
                peb REAL NOT NULL DEFAULT 0.0,
                rank INTEGER NOT NULL DEFAULT 0,
                streak INTEGER NOT NULL DEFAULT 0,
                tao_earned REAL NOT NULL DEFAULT 0.0,
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
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
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
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
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch_id INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT NOT NULL,
                api_key TEXT NOT NULL UNIQUE,
                owner TEXT NOT NULL DEFAULT '',
                tier TEXT NOT NULL DEFAULT 'free',
                request_limit INTEGER NOT NULL DEFAULT 100,
                requests_used INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_miner_epochs_uid ON miner_epochs(miner_uid);
            CREATE INDEX IF NOT EXISTS idx_miner_epochs_epoch ON miner_epochs(epoch_id);
            CREATE INDEX IF NOT EXISTS idx_task_results_epoch ON task_results(epoch_id);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_epoch ON checkpoints(epoch_id);
            CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(api_key);
        """)
        self.conn.commit()

    # ── Miner Epoch Data ────────────────────────────

    def save_miner_epoch(
        self,
        epoch_id: int,
        miner_uid: int,
        s_epoch: float,
        peb: float,
        rank: int,
        streak: int,
        tao_earned: float = 0.0,
    ) -> None:
        """Save miner performance for an epoch."""
        self.conn.execute(
            """INSERT OR REPLACE INTO miner_epochs
               (epoch_id, miner_uid, s_epoch, peb, rank, streak, tao_earned, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (epoch_id, miner_uid, s_epoch, peb, rank, streak, tao_earned, time.time()),
        )
        self.conn.commit()

    def get_miner_history(self, miner_uid: int) -> List[Dict[str, Any]]:
        """Get all epoch records for a miner."""
        rows = self.conn.execute(
            "SELECT * FROM miner_epochs WHERE miner_uid = ? ORDER BY epoch_id",
            (miner_uid,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_epoch_leaderboard(self, epoch_id: int) -> List[Dict[str, Any]]:
        """Get leaderboard for an epoch, sorted by s_epoch descending."""
        rows = self.conn.execute(
            "SELECT * FROM miner_epochs WHERE epoch_id = ? ORDER BY s_epoch DESC",
            (epoch_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ── Task Results ────────────────────────────────

    def save_task_result(
        self,
        task_id: str,
        epoch_id: int,
        domain: str,
        difficulty: int,
        is_trap: bool,
        avg_cms: float,
        best_miner_uid: Optional[int] = None,
    ) -> None:
        """Save a task result."""
        self.conn.execute(
            """INSERT INTO task_results
               (task_id, epoch_id, domain, difficulty, is_trap, avg_cms, best_miner_uid, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                epoch_id,
                domain,
                difficulty,
                int(is_trap),
                avg_cms,
                best_miner_uid,
                time.time(),
            ),
        )
        self.conn.commit()

    # ── Submissions ─────────────────────────────────

    def save_submission(
        self,
        submission_id: str,
        task_id: str,
        miner_uid: int,
        cms: float,
        quality: float,
        accuracy: float,
        novelty: float,
        efficiency: float,
        submission_hash: str = "",
    ) -> None:
        """Save a miner submission."""
        self.conn.execute(
            """INSERT INTO submissions
               (submission_id, task_id, miner_uid, cms, quality, accuracy, novelty, efficiency,
                submission_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                submission_id,
                task_id,
                miner_uid,
                cms,
                quality,
                accuracy,
                novelty,
                efficiency,
                submission_hash,
                time.time(),
            ),
        )
        self.conn.commit()

    # ── Checkpoints ─────────────────────────────────

    def save_checkpoint(self, state: Dict[str, Any], epoch_id: int) -> None:
        """Save a checkpoint as JSON."""
        self.conn.execute(
            "INSERT INTO checkpoints (epoch_id, state_json, created_at) VALUES (?, ?, ?)",
            (epoch_id, json.dumps(state), time.time()),
        )
        self.conn.commit()

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        row = self.conn.execute(
            "SELECT state_json FROM checkpoints ORDER BY epoch_id DESC, id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    def prune_checkpoints(self, keep_last: int = 5) -> None:
        """Delete old checkpoints, keeping the N most recent."""
        self.conn.execute(
            """DELETE FROM checkpoints WHERE id NOT IN (
                SELECT id FROM checkpoints ORDER BY epoch_id DESC, id DESC LIMIT ?
            )""",
            (keep_last,),
        )
        self.conn.commit()

    # ── API Keys ────────────────────────────────────

    def save_api_key(
        self,
        key_id: str,
        api_key: str,
        owner: str,
        tier: str = "free",
        request_limit: int = 100,
    ) -> None:
        """Save an API key."""
        self.conn.execute(
            """INSERT OR REPLACE INTO api_keys
               (key_id, api_key, owner, tier, request_limit, requests_used, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (key_id, api_key, owner, tier, request_limit, time.time()),
        )
        self.conn.commit()

    def get_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Look up an API key."""
        row = self.conn.execute("SELECT * FROM api_keys WHERE api_key = ?", (api_key,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def increment_api_usage(self, api_key: str) -> None:
        """Increment the usage counter for an API key."""
        self.conn.execute(
            "UPDATE api_keys SET requests_used = requests_used + 1 WHERE api_key = ?",
            (api_key,),
        )
        self.conn.commit()

    # ── Stats ───────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        stats: Dict[str, int] = {}
        for table in ("miner_epochs", "task_results", "submissions", "checkpoints", "api_keys"):
            row = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()  # noqa: S608
            stats[table] = row["cnt"] if row else 0
        return stats

    # ── Lifecycle ───────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
