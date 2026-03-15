"""
ReasonForge - Weight Setter

Computes normalized weight vector from epoch scores and submits to chain.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

logger = logging.getLogger("reasonforge.validator.weight_setter")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import bittensor as bt  # noqa: F401

    HAS_BITTENSOR = True
except ImportError:
    HAS_BITTENSOR = False


class WeightSetter:
    """Compute and submit on-chain weights from epoch scores."""

    def __init__(self, subtensor=None, wallet=None, config=None):
        self.subtensor = subtensor
        self.wallet = wallet
        self.config = config

    def compute_weights(
        self,
        miner_states: Dict[int, dict],
        n: int,
    ) -> Tuple:
        """
        Convert S_epoch + PEB into normalized weight vector.
        This is the core mapping from off-chain scoring -> on-chain Yuma Consensus input.

        Args:
            miner_states: Dict of uid -> {s_epoch, peb} data.
            n: Total number of UIDs in metagraph.

        Returns:
            Tuple of (uids_tensor, weights_tensor).
        """
        uids = []
        weights = []

        for uid in range(n):
            if uid in miner_states:
                ms = miner_states[uid]
                s_epoch = (
                    ms.get("s_epoch", 0.0) if isinstance(ms, dict) else getattr(ms, "s_epoch", 0.0)
                )
                peb = ms.get("peb", 0.0) if isinstance(ms, dict) else getattr(ms, "peb", 0.0)

                if s_epoch > 0:
                    w = s_epoch * (1.0 + peb)
                    uids.append(uid)
                    weights.append(w)

        if not weights:
            if HAS_TORCH:
                return torch.tensor([]), torch.tensor([])
            return [], []

        # Normalize
        if HAS_TORCH:
            weight_tensor = torch.FloatTensor(weights)
            weight_tensor = weight_tensor / weight_tensor.sum()
            return torch.tensor(uids), weight_tensor
        else:
            total = sum(weights)
            normalized = [w / total for w in weights]
            return uids, normalized

    def submit(
        self,
        uids,
        weights,
        netuid: int,
        max_retries: int = 3,
    ) -> bool:
        """Submit weights to chain with retry logic."""
        if not HAS_BITTENSOR or not self.subtensor or not self.wallet:
            logger.warning("Cannot submit weights: bittensor not available or not configured")
            return False

        for attempt in range(max_retries):
            try:
                success = self.subtensor.set_weights(
                    netuid=netuid,
                    wallet=self.wallet,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                if success:
                    logger.info("Weights set successfully (attempt %d)", attempt + 1)
                    return True
                else:
                    logger.warning("Weight setting returned False (attempt %d)", attempt + 1)
            except Exception as e:
                logger.warning("Weight setting attempt %d failed: %s", attempt + 1, e)
                time.sleep(5)

        logger.error("Failed to set weights after %d attempts", max_retries)
        return False
