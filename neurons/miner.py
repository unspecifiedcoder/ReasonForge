"""
ReasonForge - Miner Neuron Entry Point

Registers an Axon server, attaches handlers for each Synapse type,
and serves continuously responding to validator queries.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Tuple

# Conditional bittensor import
try:
    import bittensor as bt
    HAS_BITTENSOR = True
except ImportError:
    HAS_BITTENSOR = False

from reasonforge.base.config import MinerConfig
from reasonforge.base.neuron import BaseNeuron
from reasonforge.miner.reasoning import ReasoningEngine
from reasonforge.protocol import HealthCheck, ReasoningTask, TaskResult

logger = logging.getLogger("reasonforge.miner")


class ReasonForgeMiner(BaseNeuron):
    """Production miner neuron for the ReasonForge subnet."""

    neuron_type = "miner"

    def __init__(self, config=None):
        super().__init__(config)

        # Parse miner-specific config
        self.miner_config = MinerConfig(
            backend=getattr(self.config, "miner.backend", "openai"),
            model=getattr(self.config, "miner.model", "gpt-4o"),
            api_key_env=getattr(self.config, "miner.api_key_env", "OPENAI_API_KEY"),
            max_concurrent=getattr(self.config, "miner.max_concurrent", 4),
            port=getattr(self.config, "miner.port", 8091),
            domains=getattr(self.config, "miner.domains",
                          ["mathematics", "code", "scientific", "strategic", "causal", "ethical"]),
        )

        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(
            backend=self.miner_config.backend,
            model=self.miner_config.model,
            domains=self.miner_config.domains,
            api_key=self.miner_config.api_key,
        )

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.miner_config.max_concurrent)

        # Task tracking
        self._tasks_processed = 0
        self._last_task_time = 0.0

        # Setup Axon if bittensor is available
        self.axon = None
        if HAS_BITTENSOR and self.wallet:
            self.axon = bt.axon(wallet=self.wallet, port=self.miner_config.port)
            self.axon.attach(
                forward_fn=self.handle_reasoning_task,
                blacklist_fn=self.blacklist_reasoning_task,
                priority_fn=self.priority_reasoning_task,
            ).attach(
                forward_fn=self.handle_health_check,
            ).attach(
                forward_fn=self.handle_task_result,
            )

    async def handle_reasoning_task(self, synapse: ReasoningTask) -> ReasoningTask:
        """Core handler: receive task, produce reasoning chain, return."""
        start_time = time.time_ns()

        async with self._semaphore:
            try:
                logger.info(
                    "Processing task %s (domain=%s, difficulty=%d)",
                    synapse.task_id, synapse.domain, synapse.difficulty,
                )

                # Execute reasoning
                result = await self.reasoning_engine.solve(
                    problem=synapse.problem,
                    domain=synapse.domain,
                    difficulty=synapse.difficulty,
                    context=synapse.context,
                    constraints=synapse.constraints,
                    timeout=synapse.timeout_seconds,
                )

                # Fill mutable Synapse fields
                synapse.reasoning_steps = [
                    {
                        "step_id": i,
                        "reasoning": step.reasoning,
                        "evidence": step.evidence,
                        "confidence": step.confidence,
                        "formal_proof_fragment": step.formal_proof_fragment,
                    }
                    for i, step in enumerate(result.steps)
                ]
                synapse.final_answer = result.final_answer
                synapse.proof_status = result.proof_status
                synapse.proof_artifact = result.proof_artifact
                synapse.code_artifact = result.code_artifact
                synapse.time_taken_ms = int((time.time_ns() - start_time) / 1_000_000)
                synapse.submission_hash = synapse.compute_submission_hash()

                self._tasks_processed += 1
                self._last_task_time = time.time()

                logger.info(
                    "Task %s completed in %dms (%d steps)",
                    synapse.task_id, synapse.time_taken_ms,
                    len(synapse.reasoning_steps),
                )

            except Exception as e:
                logger.error("Task %s failed: %s", synapse.task_id, e)
                traceback.print_exc()
                synapse.final_answer = f"ERROR: {str(e)}"
                synapse.reasoning_steps = []
                synapse.time_taken_ms = int((time.time_ns() - start_time) / 1_000_000)

        return synapse

    def blacklist_reasoning_task(
        self, synapse: ReasoningTask
    ) -> Tuple[bool, str]:
        """Reject requests from non-validators or unregistered neurons."""
        if not HAS_BITTENSOR or self.metagraph is None:
            return False, ""

        try:
            caller_hotkey = synapse.dendrite.hotkey
            if caller_hotkey not in self.metagraph.hotkeys:
                return True, "Unregistered hotkey"
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
            if not self.metagraph.validator_permit[caller_uid]:
                return True, "No validator permit"
        except Exception as e:
            logger.warning("Blacklist check failed: %s", e)
            return False, ""

        return False, ""

    def priority_reasoning_task(self, synapse: ReasoningTask) -> float:
        """Higher-stake validators get priority."""
        if not HAS_BITTENSOR or self.metagraph is None:
            return 0.0

        try:
            caller_hotkey = synapse.dendrite.hotkey
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
            return float(self.metagraph.S[caller_uid])
        except Exception:
            return 0.0

    async def handle_health_check(self, synapse: HealthCheck) -> HealthCheck:
        """Respond to health checks from validators."""
        synapse.status = "ready"
        synapse.supported_domains = self.miner_config.domains
        synapse.model_info = f"{self.miner_config.backend}:{self.miner_config.model}"
        synapse.version = "0.1.0"
        return synapse

    async def handle_task_result(self, synapse: TaskResult) -> TaskResult:
        """Receive score notifications from validators (informational)."""
        if synapse.scores:
            logger.info(
                "Epoch %d results: S_epoch=%.4f, Rank=%s",
                synapse.epoch_id,
                synapse.s_epoch or 0.0,
                synapse.rank or "?",
            )
        return synapse

    def get_state_dict(self) -> dict:
        return {
            "tasks_processed": self._tasks_processed,
            "last_task_time": self._last_task_time,
            "step": self.step,
        }

    def restore_state_dict(self, state: dict) -> None:
        self._tasks_processed = state.get("tasks_processed", 0)
        self._last_task_time = state.get("last_task_time", 0.0)
        self.step = state.get("step", 0)

    def run(self) -> None:
        """Main miner loop."""
        uid_str = self.uid if self.uid is not None else "offline"
        logger.info("ReasonForge Miner starting (UID=%s)", uid_str)

        if self.axon and HAS_BITTENSOR:
            self.axon.serve(
                netuid=self.config.netuid,
                subtensor=self.subtensor,
            )
            self.axon.start()
            logger.info("Axon server started on port %d", self.miner_config.port)

        self.is_running = True

        try:
            while self.is_running:
                try:
                    if self.should_sync_metagraph():
                        self.sync()
                        self.uid = self._get_uid()

                    self.step += 1

                    # Save state periodically
                    if self.step % 100 == 0:
                        self.save_state()

                    time.sleep(12)  # One block

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error("Miner loop error: %s", e)
                    traceback.print_exc()
                    time.sleep(12)
        finally:
            if self.axon:
                self.axon.stop()
            self.save_state()
            logger.info("Miner stopped.")


def main():
    """Entry point for miner neuron."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    miner = ReasonForgeMiner()
    miner.run()


if __name__ == "__main__":
    main()
