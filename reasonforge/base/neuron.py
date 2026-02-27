"""
ReasonForge - Base Neuron Class

Shared infrastructure for miners and validators.
Handles wallet, subtensor, metagraph, registration, state persistence.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

try:
    import bittensor as bt

    HAS_BITTENSOR = True
except ImportError:
    HAS_BITTENSOR = False

from .config import create_parser

logger = logging.getLogger("reasonforge")


class BaseNeuron(ABC):
    """Shared infrastructure for miners and validators."""

    neuron_type: str = "base"

    def __init__(self, config=None):
        # 1. Parse CLI args / config
        self.config = config or self.get_config()

        # 2. Initialize Bittensor objects
        if HAS_BITTENSOR:
            self.wallet = bt.Wallet(config=self.config)
            self.subtensor = bt.Subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        else:
            self.wallet = None
            self.subtensor = None
            self.metagraph = None
            logger.warning("Bittensor not installed. Running in offline/test mode.")

        # 3. Check registration
        self.uid = self._get_uid()
        if self.uid is None and HAS_BITTENSOR:
            logger.error(
                "Neuron not registered on subnet %s. Run: btcli register --netuid %s",
                getattr(self.config, "netuid", "?"),
                getattr(self.config, "netuid", "?"),
            )

        # 4. State tracking
        self.last_sync_block: int = 0
        self.is_running: bool = False
        self.step: int = 0

        # 5. Initialize state persistence (deferred import to avoid circular)
        self.state_db = None
        self._init_state_db()

        # 6. Load previous state if exists
        self.load_state()

    def _init_state_db(self) -> None:
        """Initialize SQLite state database."""
        try:
            from ..state.database import StateDatabase

            db_dir = os.path.join("state")
            os.makedirs(db_dir, exist_ok=True)
            uid_str = self.uid if self.uid is not None else "unregistered"
            db_path = os.path.join(db_dir, f"{self.neuron_type}_{uid_str}.db")
            self.state_db = StateDatabase(db_path)
        except ImportError:
            logger.debug("State module not available, running without persistence.")
            self.state_db = None

    def get_config(self):
        """Parse CLI arguments."""
        if HAS_BITTENSOR:
            parser = create_parser(self.neuron_type)
            config = bt.Config(parser)
            return config
        else:
            parser = create_parser(self.neuron_type)
            args = parser.parse_args()
            return args

    def _get_uid(self) -> Optional[int]:
        """Find our UID in the metagraph."""
        if not HAS_BITTENSOR or self.metagraph is None or self.wallet is None:
            return None
        try:
            hotkey = self.wallet.hotkey.ss58_address
            if hotkey in self.metagraph.hotkeys:
                return self.metagraph.hotkeys.index(hotkey)
        except Exception:
            pass
        return None

    def sync(self) -> None:
        """Re-sync metagraph from chain."""
        if HAS_BITTENSOR and self.subtensor and self.metagraph:
            self.metagraph.sync(subtensor=self.subtensor)
            self.last_sync_block = self.subtensor.get_current_block()
            logger.debug("Metagraph synced at block %d", self.last_sync_block)

    def should_sync_metagraph(self) -> bool:
        """Sync every 5 blocks (~60 seconds)."""
        if not HAS_BITTENSOR or not self.subtensor:
            return False
        try:
            current_block = self.subtensor.get_current_block()
            return (current_block - self.last_sync_block) >= 5
        except Exception:
            return False

    def save_state(self) -> None:
        """Persist neuron state to SQLite."""
        if self.state_db is not None:
            try:
                state_dict = self.get_state_dict()
                self.state_db.save_checkpoint(state_dict)
            except Exception as e:
                logger.warning("Failed to save state: %s", e)

    def load_state(self) -> None:
        """Restore from last checkpoint."""
        if self.state_db is not None:
            try:
                state = self.state_db.load_latest_checkpoint()
                if state:
                    self.restore_state_dict(state)
                    logger.info("State restored from checkpoint.")
            except Exception as e:
                logger.debug("No state to restore: %s", e)

    @abstractmethod
    def get_state_dict(self) -> dict:
        """Serialize neuron state for persistence."""
        ...

    @abstractmethod
    def restore_state_dict(self, state: dict) -> None:
        """Restore neuron state from checkpoint."""
        ...

    @abstractmethod
    def run(self) -> None:
        """Main neuron loop."""
        ...

    def stop(self) -> None:
        """Signal the neuron to stop."""
        self.is_running = False
