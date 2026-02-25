"""
ReasonForge - Epoch Simulator

Full lifecycle simulation with miner profiles, validator profiles,
and the complete epoch loop (task gen -> scoring -> rewards).
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    BREAKTHROUGH_THRESHOLD,
    DIFFICULTY_MULTIPLIER,
    DOMAIN_CHECK_WEIGHTS,
    EMISSION_MINER_SHARE,
    EMISSION_VALIDATOR_SHARE,
    OBJECTIVE_WEIGHT,
    CONSENSUS_WEIGHT,
    PEB_K,
    TASKS_PER_EPOCH,
    VALIDATORS_PER_TASK,
    Domain,
    DimensionScores,
    EpochResult,
    MinerState,
    MinerSubmission,
    ReasoningStep,
    Task,
    ValidatorState,
)
from .engine import ScoringEngine
from .task_generator import TaskGenerator
from .plagiarism import PlagiarismDetector


# ──────────────────────────────────────────────
# Miner Profiles
# ──────────────────────────────────────────────

MINER_TIERS = {
    "elite":       {"q": 0.88, "a": 0.90, "n": 0.80, "e": 0.85, "var": 0.06},
    "strong":      {"q": 0.78, "a": 0.80, "n": 0.70, "e": 0.75, "var": 0.08},
    "mid":         {"q": 0.65, "a": 0.68, "n": 0.55, "e": 0.65, "var": 0.10},
    "weak":        {"q": 0.45, "a": 0.50, "n": 0.40, "e": 0.55, "var": 0.12},
    "adversarial": {"q": 0.20, "a": 0.15, "n": 0.10, "e": 0.30, "var": 0.15},
}


class MinerProfile:
    """Simulated miner with capability profile based on tier."""

    def __init__(self, miner_id: str, name: str, tier: str, seed: int = None):
        self.miner_id = miner_id
        self.name = name
        self.tier = tier
        self.rng = random.Random(seed)

        base = MINER_TIERS[tier]
        self.base_quality = base["q"]
        self.base_accuracy = base["a"]
        self.base_novelty = base["n"]
        self.base_efficiency = base["e"]
        self.variance = base["var"]

        # Random per-domain bonuses in [-0.05, 0.10]
        self.domain_bonuses: Dict[Domain, float] = {}
        for domain in Domain:
            self.domain_bonuses[domain] = self.rng.uniform(-0.05, 0.10)

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def solve_task(self, task: Task) -> Tuple[DimensionScores, MinerSubmission]:
        """
        Simulate solving a task.
        Score per dimension = base + domain_bonus - difficulty_penalty + noise
        difficulty_penalty = (difficulty - 5) * 0.015
        """
        domain_bonus = self.domain_bonuses.get(task.domain, 0.0)
        diff_penalty = (task.difficulty - 5) * 0.015

        q = self._clamp(self.base_quality + domain_bonus - diff_penalty + self.rng.gauss(0, self.variance))
        a = self._clamp(self.base_accuracy + domain_bonus - diff_penalty + self.rng.gauss(0, self.variance))
        n = self._clamp(self.base_novelty + domain_bonus - diff_penalty + self.rng.gauss(0, self.variance))
        e = self._clamp(self.base_efficiency + domain_bonus - diff_penalty + self.rng.gauss(0, self.variance))

        scores = DimensionScores(quality=q, accuracy=a, novelty=n, efficiency=e)

        # Generate simulated reasoning steps
        num_steps = self.rng.randint(2, 5)
        steps = []
        for i in range(num_steps):
            steps.append(ReasoningStep(
                step_id=i + 1,
                reasoning=f"Step {i+1}: {self.name} applies reasoning for {task.domain.value} task",
                evidence=f"Evidence from analysis of {task.problem[:50]}...",
                confidence=self._clamp(scores.cms + self.rng.gauss(0, 0.05)),
            ))

        submission = MinerSubmission(
            task_id=task.task_id,
            miner_id=self.miner_id,
            steps=steps,
            final_answer=f"Solution by {self.name} for task {task.task_id[:8]}",
            time_ms=self.rng.randint(1000, 60000),
        )
        submission.compute_hash()

        return scores, submission


# ──────────────────────────────────────────────
# Validator Profiles
# ──────────────────────────────────────────────

VALIDATOR_PROFILES = {
    "honest":    {"noise": 0.03, "bias": 0.0},
    "good":      {"noise": 0.06, "bias": 0.0},
    "lazy":      {"noise": 0.15, "bias": -0.10},
    "malicious": {"noise": 0.25, "bias": +0.20},
}


class ValidatorProfile:
    """Simulated validator with accuracy profile."""

    def __init__(self, validator_id: str, name: str, stake: float, accuracy: str, seed: int = None):
        self.validator_id = validator_id
        self.name = name
        self.stake = stake
        self.accuracy = accuracy
        self.rng = random.Random(seed)

        profile = VALIDATOR_PROFILES[accuracy]
        self.noise = profile["noise"]
        self.bias = profile["bias"]

    def evaluate(self, true_score: float) -> float:
        """Evaluate a miner's submission, adding noise and bias."""
        score = true_score + self.bias + self.rng.gauss(0, self.noise)
        return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────
# Default Roster
# ──────────────────────────────────────────────

DEFAULT_MINERS = [
    ("m-001", "DeepReason-v3", "elite"),
    ("m-002", "LogicForge-7B", "elite"),
    ("m-003", "ProofMaster", "strong"),
    ("m-004", "ReasonSwarm", "strong"),
    ("m-005", "CausalNet", "strong"),
    ("m-006", "ThinkChain", "mid"),
    ("m-007", "InferBot", "mid"),
    ("m-008", "NovaMind", "mid"),
    ("m-009", "BasicReasoner", "weak"),
    ("m-010", "CheapInference", "weak"),
    ("m-011", "SpamBot-X", "adversarial"),
    ("m-012", "CopyCat-3", "adversarial"),
]

DEFAULT_VALIDATORS = [
    ("v-001", "TruthGuard", 5000, "honest"),
    ("v-002", "AccuScore", 3000, "honest"),
    ("v-003", "FairCheck", 4000, "good"),
    ("v-004", "QuickVal", 2000, "good"),
    ("v-005", "LazyNode", 1500, "lazy"),
    ("v-006", "BadActor", 1000, "malicious"),
]


def create_default_miners(seed: int = None) -> Tuple[List[MinerProfile], List[MinerState]]:
    """Create the default roster of 12 miners."""
    profiles = []
    states = []
    for i, (mid, name, tier) in enumerate(DEFAULT_MINERS):
        s = seed + i if seed is not None else None
        profiles.append(MinerProfile(mid, name, tier, seed=s))
        states.append(MinerState(miner_id=mid, name=name))
    return profiles, states


def create_default_validators(seed: int = None) -> Tuple[List[ValidatorProfile], List[ValidatorState]]:
    """Create the default roster of 6 validators."""
    profiles = []
    states = []
    for i, (vid, name, stake, accuracy) in enumerate(DEFAULT_VALIDATORS):
        s = seed + 100 + i if seed is not None else None
        profiles.append(ValidatorProfile(vid, name, stake, accuracy, seed=s))
        states.append(ValidatorState(validator_id=vid, name=name, stake=stake))
    return profiles, states


# ──────────────────────────────────────────────
# Epoch Simulator
# ──────────────────────────────────────────────

class EpochSimulator:
    """
    Main simulation runner. Executes the full epoch loop:
    task gen -> miner solving -> scoring -> validator evaluation -> emissions.
    """

    def __init__(
        self,
        miner_profiles: List[MinerProfile],
        validator_profiles: List[ValidatorProfile],
        miner_states: Optional[List[MinerState]] = None,
        validator_states: Optional[List[ValidatorState]] = None,
        epoch_id: int = 1,
        total_emission: float = 100.0,
        seed: int = None,
    ):
        self.miner_profiles = {mp.miner_id: mp for mp in miner_profiles}
        self.validator_profiles = {vp.validator_id: vp for vp in validator_profiles}

        if miner_states is None:
            miner_states = [MinerState(miner_id=mp.miner_id, name=mp.name) for mp in miner_profiles]
        if validator_states is None:
            validator_states = [
                ValidatorState(validator_id=vp.validator_id, name=vp.name, stake=vp.stake)
                for vp in validator_profiles
            ]

        self.miner_states = {ms.miner_id: ms for ms in miner_states}
        self.validator_states = {vs.validator_id: vs for vs in validator_states}
        self.epoch_id = epoch_id
        self.total_emission = total_emission
        self.rng = random.Random(seed)
        self.task_gen = TaskGenerator(seed=seed)
        self.plagiarism = PlagiarismDetector()

    def run_epoch(self) -> EpochResult:
        """Execute the full epoch loop. Returns EpochResult."""

        # 1. Generate tasks
        tasks = self.task_gen.generate_tasks(TASKS_PER_EPOCH)

        # 2. Reset epoch accumulators
        for ms in self.miner_states.values():
            ms.epoch_scores = []
            ms.epoch_tasks = []
            ms.trap_scores = []
            ms.epoch_tao = 0.0
            ms.task_count = 0

        for vs in self.validator_states.values():
            vs.epoch_tao = 0.0
            vs.slashed_amount = 0.0

        # Track validator deviations for VAS computation
        validator_scores_given: Dict[str, List[float]] = {
            vid: [] for vid in self.validator_states
        }
        validator_consensus_ref: Dict[str, List[float]] = {
            vid: [] for vid in self.validator_states
        }

        all_cms_values: List[float] = []
        total_breakthroughs = 0
        traps_injected = sum(1 for t in tasks if t.is_trap)

        # 3. Process each task
        for task in tasks:
            task_submissions: List[MinerSubmission] = []
            task_cms_map: Dict[str, float] = {}  # miner_id -> cms for this task

            # 3a. Each miner solves the task
            miner_dim_scores: Dict[str, DimensionScores] = {}
            for mid, mprofile in self.miner_profiles.items():
                dim_scores, submission = mprofile.solve_task(task)
                task_submissions.append(submission)
                miner_dim_scores[mid] = dim_scores

            # 3b-c. For each miner, compute O_score
            for mid, dim_scores in miner_dim_scores.items():
                ms = self.miner_states[mid]

                # Compute O_score from dimension scores (simulated as quality-based)
                domain_weights = DOMAIN_CHECK_WEIGHTS.get(task.domain, {"logic": 1.0})
                # Simulate check results from dimension scores
                checks = {}
                dim_values = [dim_scores.quality, dim_scores.accuracy, dim_scores.novelty, dim_scores.efficiency]
                for i, key in enumerate(domain_weights.keys()):
                    checks[key] = dim_values[i % len(dim_values)]
                o_score = ScoringEngine.compute_objective_score(checks, domain_weights)

                # 3c-d. Assign validators and evaluate
                validator_ids = list(self.validator_states.keys())
                assigned = self.rng.sample(validator_ids, min(VALIDATORS_PER_TASK, len(validator_ids)))

                val_scores_stakes: List[Tuple[float, float]] = []
                for vid in assigned:
                    vprofile = self.validator_profiles[vid]
                    v_score = vprofile.evaluate(o_score)
                    val_scores_stakes.append((v_score, vprofile.stake))

                # 3e. Compute C_score via consensus
                c_score = ScoringEngine.compute_consensus_score(val_scores_stakes)

                # 3f. FinalScore
                final_score = ScoringEngine.compute_final_score(o_score, c_score)

                # 3g. Map FinalScore back to dimension scores
                # Scale original dimensions proportionally
                original_cms = dim_scores.cms
                if original_cms > 0:
                    scale = final_score / original_cms
                else:
                    scale = 1.0
                adjusted = DimensionScores(
                    quality=max(0, min(1, dim_scores.quality * scale)),
                    accuracy=max(0, min(1, dim_scores.accuracy * scale)),
                    novelty=max(0, min(1, dim_scores.novelty * scale)),
                    efficiency=max(0, min(1, dim_scores.efficiency * scale)),
                )

                # 3h. Compute CMS
                cms = ScoringEngine.compute_cms(adjusted)

                # 3i. Apply breakthrough multiplier if applicable
                is_breakthrough = task.previously_unsolved and cms > BREAKTHROUGH_THRESHOLD
                if is_breakthrough:
                    cms = ScoringEngine.apply_breakthrough(cms, True)
                    total_breakthroughs += 1
                    ms.breakthroughs += 1

                # 3j. Track VAS deviations for each assigned validator
                for vid_idx, vid in enumerate(assigned):
                    v_given = val_scores_stakes[vid_idx][0]
                    validator_scores_given[vid].append(v_given)
                    validator_consensus_ref[vid].append(c_score)

                # 3k. Store CMS in miner's epoch_scores
                ms.epoch_scores.append(cms)
                ms.epoch_tasks.append(task.task_id)
                ms.task_count += 1
                task_cms_map[mid] = cms
                all_cms_values.append(cms)

                # 3l. If trap task, store in trap_scores
                if task.is_trap:
                    ms.trap_scores.append(cms)

        # 4. Compute S_epoch for each miner
        for mid, ms in self.miner_states.items():
            if ms.epoch_scores:
                diff_mults = []
                for tid in ms.epoch_tasks:
                    task_obj = next((t for t in tasks if t.task_id == tid), None)
                    diff_mults.append(task_obj.difficulty_multiplier if task_obj else 1.0)

                trap_penalty = ScoringEngine.compute_trap_penalty(ms.trap_scores)
                ms.s_epoch = ScoringEngine.compute_s_epoch(
                    ms.epoch_scores, diff_mults, trap_penalty
                )
            else:
                ms.s_epoch = 0.0

        # 5. Rank miners by S_epoch descending
        sorted_miners = sorted(
            self.miner_states.values(), key=lambda m: m.s_epoch, reverse=True
        )
        for rank, ms in enumerate(sorted_miners, 1):
            ms.rank = rank
            # Update streak: increment if in top-K, else reset
            if rank <= PEB_K and ms.s_epoch > 0:
                ms.streak += 1
            else:
                ms.streak = 0

        # 6. Compute PEB for top-K
        for ms in self.miner_states.values():
            ms.peb = ScoringEngine.compute_peb(ms.rank, ms.streak)

        # 7. Distribute miner emissions (Eq. 5)
        miner_pool = self.total_emission * EMISSION_MINER_SHARE
        miner_list = sorted_miners  # already sorted
        rewards = ScoringEngine.distribute_miner_emissions(miner_list, miner_pool)
        for ms, reward in zip(miner_list, rewards):
            ms.epoch_tao = round(reward, 6)
            ms.total_tao_earned += ms.epoch_tao

        # 8. Finalize validator VAS
        for vid, vs in self.validator_states.items():
            v_scores = validator_scores_given.get(vid, [])
            c_scores = validator_consensus_ref.get(vid, [])
            if v_scores:
                vs.current_vas = ScoringEngine.compute_vas(v_scores, c_scores)
            else:
                vs.current_vas = 1.0
            vs.vas_history.append(vs.current_vas)
            vs.evaluations_count += len(v_scores)

        # 9. Compute validator reputation multipliers
        for vs in self.validator_states.values():
            vs.compute_reputation_multiplier()

        # 10. Compute slashing for underperformers
        for vs in self.validator_states.values():
            vs.compute_slash()

        # 11. Distribute validator emissions (Eq. 8)
        validator_pool = self.total_emission * EMISSION_VALIDATOR_SHARE
        val_list = list(self.validator_states.values())
        val_rewards = ScoringEngine.distribute_validator_emissions(val_list, validator_pool)
        for vs, reward in zip(val_list, val_rewards):
            vs.epoch_tao = round(reward, 6)
            vs.total_tao_earned += vs.epoch_tao

        # 12. Build and return EpochResult
        miner_results = []
        for ms in sorted_miners:
            miner_results.append({
                "rank": ms.rank,
                "miner_id": ms.miner_id,
                "name": ms.name,
                "s_epoch": round(ms.s_epoch, 6),
                "peb": round(ms.peb, 6),
                "streak": ms.streak,
                "epoch_tao": ms.epoch_tao,
                "total_tao": round(ms.total_tao_earned, 6),
                "trap_penalty": round(ms.trap_penalty, 6),
                "breakthroughs": ms.breakthroughs,
            })

        validator_results = []
        for vs in val_list:
            validator_results.append({
                "validator_id": vs.validator_id,
                "name": vs.name,
                "stake": vs.stake,
                "vas": round(vs.current_vas, 6),
                "reputation": round(vs.reputation_multiplier, 6),
                "epoch_tao": vs.epoch_tao,
                "total_tao": round(vs.total_tao_earned, 6),
                "slashed": round(vs.slashed_amount, 6),
            })

        avg_cms = sum(all_cms_values) / len(all_cms_values) if all_cms_values else 0.0

        result = EpochResult(
            epoch_id=self.epoch_id,
            total_emission=self.total_emission,
            miner_pool=round(miner_pool, 6),
            validator_pool=round(validator_pool, 6),
            miner_results=miner_results,
            validator_results=validator_results,
            tasks_processed=len(tasks),
            traps_injected=traps_injected,
            breakthroughs=total_breakthroughs,
            avg_cms=round(avg_cms, 6),
        )

        return result

    @staticmethod
    def to_json(result: EpochResult) -> Dict[str, Any]:
        """Serialize EpochResult to JSON-safe dict."""
        return {
            "epoch_id": result.epoch_id,
            "total_emission": result.total_emission,
            "miner_pool": result.miner_pool,
            "validator_pool": result.validator_pool,
            "tasks_processed": result.tasks_processed,
            "traps_injected": result.traps_injected,
            "breakthroughs": result.breakthroughs,
            "avg_cms": result.avg_cms,
            "miners": result.miner_results,
            "validators": result.validator_results,
            "timestamp": result.timestamp,
        }
