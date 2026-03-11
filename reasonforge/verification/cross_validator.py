"""Cross-validator: consensus across multiple miner verdicts."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .verdict import StepVerdict, VerificationVerdict


class CrossValidator:
    """Build a consensus verdict from multiple independent miner results."""

    def cross_validate(
        self,
        miner_verdicts: Dict[int, Optional[VerificationVerdict]],
    ) -> VerificationVerdict:
        """Merge verdicts produced by different miners into a single consensus.

        For every step_id seen across the verdicts the step is marked
        *verified* when a strict majority (> N/2) of **valid** miners
        (those that supplied a non-``None`` verdict) marked it as verified.

        Parameters
        ----------
        miner_verdicts:
            Mapping of miner UID to the verdict it produced.  ``None``
            entries are ignored (miner did not respond).

        Returns
        -------
        VerificationVerdict
            A consensus verdict whose ``translator_uids`` records the UIDs
            of all miners that contributed.
        """
        valid_verdicts: Dict[int, VerificationVerdict] = {
            uid: v for uid, v in miner_verdicts.items() if v is not None
        }
        n_valid = len(valid_verdicts)

        if n_valid == 0:
            return VerificationVerdict(
                task_id="consensus",
                overall="FAILED",
                raw_output="No valid miner verdicts received.",
            )

        # Collect task_id from the first available verdict.
        task_id = next(iter(valid_verdicts.values())).task_id

        # Tally per-step verification counts.
        step_yes: Dict[int, int] = defaultdict(int)
        step_total: Dict[int, int] = defaultdict(int)
        step_formal: Dict[int, str] = {}

        for _uid, verdict in valid_verdicts.items():
            for sv in verdict.step_verdicts:
                step_total[sv.step_id] += 1
                if sv.verified:
                    step_yes[sv.step_id] += 1
                # Keep the latest non-empty formal representation.
                if sv.formal_representation:
                    step_formal[sv.step_id] = sv.formal_representation

        # Build consensus step verdicts.
        threshold = n_valid / 2.0
        consensus_steps: List[StepVerdict] = []

        for step_id in sorted(step_total.keys()):
            verified = step_yes.get(step_id, 0) > threshold
            consensus_steps.append(
                StepVerdict(
                    step_id=step_id,
                    verified=verified,
                    formal_representation=step_formal.get(step_id, ""),
                    details={
                        "yes_votes": step_yes.get(step_id, 0),
                        "total_votes": step_total[step_id],
                        "n_valid_miners": n_valid,
                    },
                )
            )

        verified_count = sum(1 for sv in consensus_steps if sv.verified)
        failure_points = [sv for sv in consensus_steps if not sv.verified]
        overall = (
            "PASSED"
            if verified_count == len(consensus_steps) and consensus_steps
            else "FAILED"
        )

        return VerificationVerdict(
            task_id=task_id,
            overall=overall,
            step_verdicts=consensus_steps,
            total_steps=len(consensus_steps),
            verified_steps=verified_count,
            failure_points=failure_points,
            translator_uids=sorted(valid_verdicts.keys()),
        )
