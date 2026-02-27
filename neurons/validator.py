"""
ReasonForge - Validator Neuron Entry Point

Runs the main epoch loop: generate tasks, query miners, score responses,
compute and set on-chain weights.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import bittensor as bt
    HAS_BITTENSOR = True
except ImportError:
    HAS_BITTENSOR = False

from reasonforge.base.config import ValidatorConfig
from reasonforge.base.neuron import BaseNeuron
from reasonforge.engine import ScoringEngine
from reasonforge.protocol import ReasoningTask, verify_submission_hash
from reasonforge.types import (
    PEB_K,
    SIMILARITY_PENALTY,
    SIMILARITY_THRESHOLD,
    DimensionScores,
    MinerState,
    Task,
)
from reasonforge.validator.scoring import ValidatorScorer
from reasonforge.validator.task_manager import TaskManager
from reasonforge.validator.trap_manager import TrapManager
from reasonforge.validator.weight_setter import WeightSetter

logger = logging.getLogger("reasonforge.validator")


@dataclass
class TaskProcessingResult:
    """Result of processing a single task across all miners."""
    task: Task
    scored_results: List[Tuple[int, DimensionScores]] = field(default_factory=list)


class ReasonForgeValidator(BaseNeuron):
    """Production validator neuron for the ReasonForge subnet."""

    neuron_type = "validator"

    def __init__(self, config=None):
        super().__init__(config)

        # Parse validator-specific config
        self.val_config = ValidatorConfig(
            epoch_length=getattr(self.config, "validator.epoch_length", 360),
            tasks_per_epoch=getattr(self.config, "validator.tasks_per_epoch", 12),
            trap_rate=getattr(self.config, "validator.trap_rate", 0.15),
            timeout=getattr(self.config, "validator.timeout", 300),
            sample_size=getattr(self.config, "validator.sample_size", 16),
            sandbox_enabled=getattr(self.config, "validator.sandbox_enabled", False),
            lean4_enabled=getattr(self.config, "validator.lean4_enabled", False),
            embedding_model=getattr(self.config, "validator.embedding_model", "all-MiniLM-L6-v2"),
        )

        # Initialize components
        self.dendrite = None
        if HAS_BITTENSOR and self.wallet:
            self.dendrite = bt.dendrite(wallet=self.wallet)

        self.task_manager = TaskManager()
        self.trap_manager = TrapManager(trap_rate=self.val_config.trap_rate)
        self.scorer = ValidatorScorer(
            lean4_enabled=self.val_config.lean4_enabled,
            sandbox_enabled=self.val_config.sandbox_enabled,
        )
        self.weight_setter = WeightSetter(
            subtensor=self.subtensor,
            wallet=self.wallet,
            config=self.config,
        )

        # Similarity detector (lazy-loaded)
        self.similarity_detector = None

        # State tracking
        self.miner_states: Dict[int, MinerState] = {}
        self.epoch_id: int = 0
        self.last_epoch_block: int = 0
        self.scores = None
        if HAS_TORCH:
            self.scores = torch.zeros(256)

    def _get_similarity_detector(self):
        """Lazy-load similarity detector to avoid import cost at startup."""
        if self.similarity_detector is None:
            try:
                from reasonforge.embeddings.similarity import SimilarityDetector
                self.similarity_detector = SimilarityDetector(
                    model_name=self.val_config.embedding_model
                )
            except ImportError:
                logger.warning("Embedding similarity not available, using basic detection")
        return self.similarity_detector

    def is_epoch_boundary(self, current_block: int) -> bool:
        """Check if we've reached the next epoch boundary."""
        return (current_block - self.last_epoch_block) >= self.val_config.epoch_length

    def get_queryable_miners(self) -> List[int]:
        """Get list of miner UIDs to query."""
        if not HAS_BITTENSOR or self.metagraph is None:
            return list(range(min(10, 256)))  # Test mode

        miner_uids = []
        for uid in range(self.metagraph.n):
            if uid == self.uid:
                continue  # Skip self
            # Check if axon is serving
            axon = self.metagraph.axons[uid]
            if axon.ip != "0.0.0.0" and axon.port > 0:
                miner_uids.append(uid)

        # Sample if too many
        if len(miner_uids) > self.val_config.sample_size:
            import random
            miner_uids = random.sample(miner_uids, self.val_config.sample_size)

        return miner_uids

    def get_or_create_miner_state(self, uid: int) -> MinerState:
        """Get or create miner state for a UID."""
        if uid not in self.miner_states:
            self.miner_states[uid] = MinerState(miner_id=str(uid))
        return self.miner_states[uid]

    def run(self) -> None:
        """Main validator loop."""
        uid_str = self.uid if self.uid is not None else "offline"
        logger.info("ReasonForge Validator starting (UID=%s)", uid_str)

        self.is_running = True

        try:
            while self.is_running:
                try:
                    # 1. Sync metagraph
                    if self.should_sync_metagraph():
                        self.sync()

                    # 2. Check if epoch boundary
                    if HAS_BITTENSOR and self.subtensor:
                        current_block = self.subtensor.get_current_block()
                        if self.is_epoch_boundary(current_block):
                            self._run_epoch()
                            self.last_epoch_block = current_block
                    else:
                        # Offline mode: run epoch every 60 seconds
                        self._run_epoch()
                        time.sleep(60)
                        continue

                    # 3. Sleep for one block
                    time.sleep(12)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error("Validator loop error: %s", e)
                    traceback.print_exc()
                    time.sleep(12)
        finally:
            self.save_state()
            logger.info("Validator stopped.")

    def _run_epoch(self) -> None:
        """Execute one complete scoring epoch."""
        self.epoch_id += 1
        logger.info("=== EPOCH %d ===", self.epoch_id)

        # Phase A: Generate tasks
        tasks = self.task_manager.generate_epoch_tasks(
            count=self.val_config.tasks_per_epoch,
            trap_rate=self.val_config.trap_rate,
        )
        logger.info("Generated %d tasks (%d traps)",
                     len(tasks), sum(1 for t in tasks if t.is_trap))

        # Phase B: Process each task
        all_task_results = []
        for task in tasks:
            result = asyncio.get_event_loop().run_until_complete(
                self._process_task(task)
            )
            all_task_results.append(result)

        # Phase C: Compute epoch scores
        self._compute_epoch_scores(all_task_results)

        # Phase D: Set on-chain weights
        self._set_weights()

        # Phase E: Persist state
        self.save_state()

        # Phase F: Log results
        self._log_epoch_results()

    async def _process_task(self, task: Task) -> TaskProcessingResult:
        """Query miners, collect responses, score them."""
        miner_uids = self.get_queryable_miners()

        if not miner_uids:
            logger.warning("No queryable miners found")
            return TaskProcessingResult(task=task)

        # Build synapse
        from reasonforge.protocol import create_reasoning_task
        synapse = create_reasoning_task(
            task_id=task.task_id,
            problem=task.problem,
            domain=task.domain.value if hasattr(task.domain, "value") else task.domain,
            difficulty=task.difficulty,
            timeout_seconds=self.val_config.timeout,
        )

        # Query miners
        responses = []
        if self.dendrite and HAS_BITTENSOR and self.metagraph:
            axons = [self.metagraph.axons[uid] for uid in miner_uids]
            responses = await self.dendrite(
                axons=axons,
                synapse=synapse,
                timeout=self.val_config.timeout,
            )
        else:
            # Test mode: empty responses
            responses = [ReasoningTask(**synapse.model_dump()) for _ in miner_uids]

        # Score each response
        scored_results = []
        for uid, response in zip(miner_uids, responses):
            try:
                # Check for timeout/failure
                if response.final_answer is None:
                    scored_results.append((uid, DimensionScores(0, 0, 0, 0)))
                    continue

                # Verify submission hash
                if response.submission_hash and not verify_submission_hash(response):
                    logger.warning("UID %d: hash mismatch, penalizing", uid)
                    scored_results.append((uid, DimensionScores(0, 0, 0, 0)))
                    continue

                # Compute dimension scores
                response_data = response.deserialize()
                dim_scores = await self.scorer.compute_dimensions(task, response_data)

                # Check plagiarism
                sim_detector = self._get_similarity_detector()
                if sim_detector:
                    try:
                        similarity = sim_detector.check_against_batch(
                            response, [r for r in responses if r != response]
                        )
                        if similarity > SIMILARITY_THRESHOLD:
                            dim_scores = DimensionScores(
                                quality=dim_scores.quality * SIMILARITY_PENALTY,
                                accuracy=dim_scores.accuracy * SIMILARITY_PENALTY,
                                novelty=dim_scores.novelty * SIMILARITY_PENALTY,
                                efficiency=dim_scores.efficiency,
                            )
                    except Exception as e:
                        logger.debug("Similarity check failed: %s", e)

                scored_results.append((uid, dim_scores))

                # Track trap scores
                if task.is_trap:
                    trap_score = self.trap_manager.evaluate_trap_response(
                        task, response.final_answer, response.reasoning_steps
                    )
                    self.trap_manager.record_trap_score(uid, trap_score)

            except Exception as e:
                logger.warning("Scoring UID %d failed: %s", uid, e)
                scored_results.append((uid, DimensionScores(0, 0, 0, 0)))

        return TaskProcessingResult(task=task, scored_results=scored_results)

    def _compute_epoch_scores(self, task_results: List[TaskProcessingResult]) -> None:
        """Aggregate per-task CMS into S_epoch using MVP engine."""
        # Collect per-miner scores
        miner_task_scores: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        for tr in task_results:
            for uid, dim_scores in tr.scored_results:
                cms = ScoringEngine.compute_cms(dim_scores)
                diff_mult = tr.task.difficulty_multiplier
                miner_task_scores[uid].append((cms, diff_mult))

        # Compute S_epoch for each miner
        for uid, scores in miner_task_scores.items():
            ms = self.get_or_create_miner_state(uid)
            cms_list = [s[0] for s in scores]
            diff_mults = [s[1] for s in scores]

            trap_penalty = self.trap_manager.get_trap_penalty(uid)

            ms.s_epoch = ScoringEngine.compute_s_epoch(
                cms_list, diff_mults, trap_penalty
            )
            ms.epoch_scores.append(ms.s_epoch)
            ms.task_count += len(cms_list)

        # Rank miners
        ranked = sorted(
            [(uid, ms) for uid, ms in self.miner_states.items() if ms.s_epoch > 0],
            key=lambda x: x[1].s_epoch, reverse=True,
        )

        for i, (uid, ms) in enumerate(ranked):
            ms.rank = i + 1
            if ms.rank <= PEB_K:
                ms.streak += 1
            else:
                ms.streak = 0
            ms.peb = ScoringEngine.compute_peb(ms.rank, ms.streak)

    def _set_weights(self) -> None:
        """Compute and set on-chain weights."""
        n = 256
        if HAS_BITTENSOR and self.metagraph:
            n = self.metagraph.n

        miner_data = {}
        for uid, ms in self.miner_states.items():
            miner_data[uid] = {"s_epoch": ms.s_epoch, "peb": ms.peb}

        uids, weights = self.weight_setter.compute_weights(miner_data, n)

        if HAS_BITTENSOR and self.subtensor:
            netuid = getattr(self.config, "netuid", 1)
            success = self.weight_setter.submit(uids, weights, netuid)
            if success:
                logger.info("Weights set for epoch %d", self.epoch_id)
            else:
                logger.error("Failed to set weights for epoch %d", self.epoch_id)
        else:
            logger.info("Weights computed (offline mode): %d non-zero entries",
                        len(uids) if hasattr(uids, '__len__') else 0)

    def _log_epoch_results(self) -> None:
        """Log epoch results summary."""
        active = [(uid, ms) for uid, ms in self.miner_states.items() if ms.s_epoch > 0]
        if not active:
            logger.info("Epoch %d: No active miners", self.epoch_id)
            return

        active.sort(key=lambda x: x[1].s_epoch, reverse=True)
        logger.info("Epoch %d results (%d active miners):", self.epoch_id, len(active))
        for uid, ms in active[:10]:
            logger.info(
                "  UID %d: S_epoch=%.4f, PEB=%.4f, Rank=%d, Streak=%d",
                uid, ms.s_epoch, ms.peb, ms.rank, ms.streak,
            )

    def get_state_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "last_epoch_block": self.last_epoch_block,
            "miner_states": {
                uid: {
                    "s_epoch": ms.s_epoch,
                    "peb": ms.peb,
                    "rank": ms.rank,
                    "streak": ms.streak,
                    "total_tao_earned": ms.total_tao_earned,
                    "task_count": ms.task_count,
                }
                for uid, ms in self.miner_states.items()
            },
        }

    def restore_state_dict(self, state: dict) -> None:
        self.epoch_id = state.get("epoch_id", 0)
        self.last_epoch_block = state.get("last_epoch_block", 0)
        for uid_str, ms_data in state.get("miner_states", {}).items():
            uid = int(uid_str)
            ms = self.get_or_create_miner_state(uid)
            ms.s_epoch = ms_data.get("s_epoch", 0.0)
            ms.peb = ms_data.get("peb", 0.0)
            ms.rank = ms_data.get("rank", 0)
            ms.streak = ms_data.get("streak", 0)
            ms.total_tao_earned = ms_data.get("total_tao_earned", 0.0)
            ms.task_count = ms_data.get("task_count", 0)


def main():
    """Entry point for validator neuron."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    validator = ReasonForgeValidator()
    validator.run()


if __name__ == "__main__":
    main()
