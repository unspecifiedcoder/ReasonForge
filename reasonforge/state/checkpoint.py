"""
ReasonForge - Checkpoint Manager

Higher-level checkpoint save/restore built on top of StateDatabase.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .database import StateDatabase

logger = logging.getLogger("reasonforge.state.checkpoint")


class CheckpointManager:
    """Manage checkpoint save/restore operations."""

    def __init__(self, db: StateDatabase, keep_last: int = 10):
        self.db = db
        self.keep_last = keep_last

    def save(self, epoch_id: int, state: Dict[str, Any]) -> None:
        """Save a checkpoint and optionally prune old ones."""
        self.db.save_checkpoint(state, epoch_id=epoch_id)
        self.db.prune_checkpoints(keep_last=self.keep_last)
        logger.info("Checkpoint saved for epoch %d", epoch_id)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        return self.db.load_latest_checkpoint()
