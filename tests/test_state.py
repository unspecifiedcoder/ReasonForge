"""
ReasonForge - State Persistence Tests

Tests for SQLite database, checkpoints, and migrations.
"""

import pytest

from reasonforge.state.checkpoint import CheckpointManager
from reasonforge.state.database import StateDatabase
from reasonforge.state.migrations import MigrationManager


class TestStateDatabase:
    """Test SQLite state persistence."""

    @pytest.fixture
    def db(self, temp_db_path):
        db = StateDatabase(temp_db_path)
        yield db
        db.close()

    def test_init_creates_tables(self, db):
        stats = db.get_stats()
        assert "miner_epochs" in stats
        assert "task_results" in stats
        assert "checkpoints" in stats

    def test_save_and_load_miner_epoch(self, db):
        db.save_miner_epoch(
            epoch_id=1,
            miner_uid=42,
            s_epoch=0.85,
            peb=0.12,
            rank=3,
            streak=5,
            tao_earned=1.5,
        )
        history = db.get_miner_history(42)
        assert len(history) == 1
        assert history[0]["s_epoch"] == 0.85
        assert history[0]["rank"] == 3

    def test_multiple_epochs(self, db):
        for i in range(5):
            db.save_miner_epoch(
                epoch_id=i,
                miner_uid=1,
                s_epoch=0.5 + i * 0.1,
                peb=0.0,
                rank=1,
                streak=i,
            )
        history = db.get_miner_history(1)
        assert len(history) == 5

    def test_epoch_leaderboard(self, db):
        db.save_miner_epoch(epoch_id=1, miner_uid=1, s_epoch=0.9, peb=0.1, rank=1, streak=3)
        db.save_miner_epoch(epoch_id=1, miner_uid=2, s_epoch=0.7, peb=0.0, rank=2, streak=1)
        db.save_miner_epoch(epoch_id=1, miner_uid=3, s_epoch=0.5, peb=0.0, rank=3, streak=0)

        board = db.get_epoch_leaderboard(1)
        assert len(board) == 3
        assert board[0]["miner_uid"] == 1  # Highest score first
        assert board[0]["s_epoch"] > board[1]["s_epoch"]

    def test_save_and_load_checkpoint(self, db):
        state = {"epoch_id": 5, "scores": [0.8, 0.7, 0.6]}
        db.save_checkpoint(state, epoch_id=5)

        loaded = db.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["epoch_id"] == 5
        assert len(loaded["scores"]) == 3

    def test_checkpoint_latest(self, db):
        db.save_checkpoint({"epoch": 1}, epoch_id=1)
        db.save_checkpoint({"epoch": 2}, epoch_id=2)
        db.save_checkpoint({"epoch": 3}, epoch_id=3)

        loaded = db.load_latest_checkpoint()
        assert loaded["epoch"] == 3

    def test_prune_checkpoints(self, db):
        for i in range(20):
            db.save_checkpoint({"epoch": i}, epoch_id=i)

        db.prune_checkpoints(keep_last=5)

        # Should still load the latest
        loaded = db.load_latest_checkpoint()
        assert loaded["epoch"] == 19

    def test_save_task_result(self, db):
        db.save_task_result(
            task_id="t001",
            epoch_id=1,
            domain="mathematics",
            difficulty=5,
            is_trap=False,
            avg_cms=0.75,
            best_miner_uid=1,
        )
        stats = db.get_stats()
        assert stats["task_results"] == 1

    def test_save_submission(self, db):
        db.save_submission(
            submission_id="s001",
            task_id="t001",
            miner_uid=1,
            cms=0.8,
            quality=0.9,
            accuracy=0.7,
            novelty=0.6,
            efficiency=0.8,
            submission_hash="abc123",
        )
        stats = db.get_stats()
        assert stats["submissions"] == 1

    def test_api_key_crud(self, db):
        db.save_api_key("k1", "rf_test_key_123", "test_user", "free", 100)
        key = db.get_api_key("rf_test_key_123")
        assert key is not None
        assert key["owner"] == "test_user"
        assert key["requests_used"] == 0

        db.increment_api_usage("rf_test_key_123")
        key = db.get_api_key("rf_test_key_123")
        assert key["requests_used"] == 1


class TestCheckpointManager:
    """Test checkpoint save/restore."""

    @pytest.fixture
    def manager(self, temp_db_path):
        db = StateDatabase(temp_db_path)
        mgr = CheckpointManager(db)
        yield mgr
        db.close()

    def test_save_and_load(self, manager):
        state = {"epoch_id": 10, "miners": {"1": {"score": 0.8}}}
        manager.save(epoch_id=10, state=state)

        loaded = manager.load_latest()
        assert loaded is not None
        assert loaded["epoch_id"] == 10

    def test_load_empty(self, manager):
        loaded = manager.load_latest()
        assert loaded is None


class TestMigrations:
    """Test database migrations."""

    @pytest.fixture
    def db_conn(self, temp_db_path):
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        yield conn
        conn.close()

    def test_initial_version(self, db_conn):
        mgr = MigrationManager(db_conn)
        assert mgr.get_current_version() == 0

    def test_apply_migrations(self, db_conn):
        mgr = MigrationManager(db_conn)
        applied = mgr.apply_pending()
        assert applied > 0
        assert mgr.get_current_version() > 0

    def test_idempotent(self, db_conn):
        mgr = MigrationManager(db_conn)
        mgr.apply_pending()
        v1 = mgr.get_current_version()
        mgr.apply_pending()
        v2 = mgr.get_current_version()
        assert v1 == v2
