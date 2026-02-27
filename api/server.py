"""
ReasonForge - FastAPI Server

REST API serving simulation data to the dashboard.
Run with: uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

# Add parent directory to path so we can import reasonforge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from reasonforge.simulator import (
    EpochSimulator,
    create_default_miners,
    create_default_validators,
)
from reasonforge.types import (
    BREAKTHROUGH_MULTIPLIER,
    BREAKTHROUGH_THRESHOLD,
    CONSENSUS_TRIM_DELTA,
    CONSENSUS_WEIGHT,
    DIFFICULTY_MULTIPLIER,
    DOMAIN_CHECK_WEIGHTS,
    EMISSION_MINER_SHARE,
    EMISSION_VALIDATOR_SHARE,
    OBJECTIVE_WEIGHT,
    PEB_ALPHA,
    PEB_K,
    PEB_STREAK_CAP,
    SIMILARITY_PENALTY,
    SIMILARITY_THRESHOLD,
    TASKS_PER_EPOCH,
    TRAP_RATE,
    TRAP_THRESHOLD,
    VALIDATORS_PER_TASK,
    VAS_REP_MAX_MULTIPLIER,
    VAS_REP_THRESHOLD,
    VAS_SLASH_GAMMA,
    VAS_SLASH_THRESHOLD,
    W_ACCURACY,
    W_EFFICIENCY,
    W_NOVELTY,
    W_QUALITY,
)

app = FastAPI(
    title="ReasonForge API",
    description="REST API for the ReasonForge decentralized reasoning subnet simulator",
    version="0.1.0",
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────

class SimulateRequest(BaseModel):
    epochs: int = 5
    emission: float = 100.0
    seed: Optional[int] = 42


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/simulate")
async def simulate(request: SimulateRequest):
    """Run N epochs and return full results JSON."""
    seed = request.seed
    miner_profiles, miner_states = create_default_miners(seed=seed)
    validator_profiles, validator_states = create_default_validators(seed=seed)

    all_epochs = []
    for epoch in range(1, request.epochs + 1):
        epoch_seed = (seed + epoch * 1000) if seed is not None else None
        sim = EpochSimulator(
            miner_profiles=miner_profiles,
            validator_profiles=validator_profiles,
            miner_states=miner_states,
            validator_states=validator_states,
            epoch_id=epoch,
            total_emission=request.emission,
            seed=epoch_seed,
        )
        result = sim.run_epoch()
        all_epochs.append(EpochSimulator.to_json(result))

    return {
        "config": {
            "epochs": request.epochs,
            "emission": request.emission,
            "miners": len(miner_profiles),
            "validators": len(validator_profiles),
        },
        "epochs": all_epochs,
    }


@app.get("/api/simulate/stream")
async def simulate_stream(
    epochs: int = Query(default=10, ge=1, le=100),
    emission: float = Query(default=100.0, gt=0),
    seed: Optional[int] = Query(default=42),
):
    """SSE stream, emitting one epoch at a time for live dashboard."""

    async def event_generator():
        s = seed
        miner_profiles, miner_states = create_default_miners(seed=s)
        validator_profiles, validator_states = create_default_validators(seed=s)

        for epoch in range(1, epochs + 1):
            epoch_seed = (s + epoch * 1000) if s is not None else None
            sim = EpochSimulator(
                miner_profiles=miner_profiles,
                validator_profiles=validator_profiles,
                miner_states=miner_states,
                validator_states=validator_states,
                epoch_id=epoch,
                total_emission=emission,
                seed=epoch_seed,
            )
            result = sim.run_epoch()
            data = json.dumps(EpochSimulator.to_json(result))
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.1)  # Small delay between epochs

        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/constants")
async def constants():
    """Return all protocol constants."""
    return {
        "cms_weights": {
            "W_QUALITY": W_QUALITY,
            "W_ACCURACY": W_ACCURACY,
            "W_NOVELTY": W_NOVELTY,
            "W_EFFICIENCY": W_EFFICIENCY,
        },
        "emission_split": {
            "EMISSION_MINER_SHARE": EMISSION_MINER_SHARE,
            "EMISSION_VALIDATOR_SHARE": EMISSION_VALIDATOR_SHARE,
        },
        "peb": {
            "PEB_ALPHA": PEB_ALPHA,
            "PEB_K": PEB_K,
            "PEB_STREAK_CAP": PEB_STREAK_CAP,
        },
        "breakthrough": {
            "BREAKTHROUGH_MULTIPLIER": BREAKTHROUGH_MULTIPLIER,
            "BREAKTHROUGH_THRESHOLD": BREAKTHROUGH_THRESHOLD,
        },
        "trap": {
            "TRAP_RATE": TRAP_RATE,
            "TRAP_THRESHOLD": TRAP_THRESHOLD,
        },
        "similarity": {
            "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD,
            "SIMILARITY_PENALTY": SIMILARITY_PENALTY,
        },
        "validator_slashing": {
            "VAS_SLASH_THRESHOLD": VAS_SLASH_THRESHOLD,
            "VAS_SLASH_GAMMA": VAS_SLASH_GAMMA,
        },
        "validator_reputation": {
            "VAS_REP_THRESHOLD": VAS_REP_THRESHOLD,
            "VAS_REP_MAX_MULTIPLIER": VAS_REP_MAX_MULTIPLIER,
        },
        "operational": {
            "TASKS_PER_EPOCH": TASKS_PER_EPOCH,
            "VALIDATORS_PER_TASK": VALIDATORS_PER_TASK,
        },
        "scoring": {
            "OBJECTIVE_WEIGHT": OBJECTIVE_WEIGHT,
            "CONSENSUS_WEIGHT": CONSENSUS_WEIGHT,
            "CONSENSUS_TRIM_DELTA": CONSENSUS_TRIM_DELTA,
        },
        "difficulty_multiplier": DIFFICULTY_MULTIPLIER,
        "domain_check_weights": {
            d.value: w for d, w in DOMAIN_CHECK_WEIGHTS.items()
        },
    }


@app.get("/api/formulas")
async def formulas():
    """Return formula descriptions as JSON."""
    return {
        "formulas": [
            {
                "id": "eq1",
                "name": "Emission Split",
                "latex": "E_total = E_miner + E_validator = 0.90*E + 0.10*E",
                "description": "Total emissions split between miners (90%) and validators (10%)",
            },
            {
                "id": "eq2",
                "name": "Composite Miner Score (CMS)",
                "latex": "CMS(m,t) = 0.40*Q + 0.30*A + 0.15*N + 0.15*Eff",
                "description": "Weighted combination of quality, accuracy, novelty, and efficiency",
            },
            {
                "id": "eq3",
                "name": "Epoch Score",
                "latex": "S_epoch(m) = (1/|T_m|) * sum(CMS * D(t)) * trap_penalty",
                "description": "Difficulty-weighted average CMS with trap penalty",
            },
            {
                "id": "eq4",
                "name": "Persistent Excellence Bonus",
                "latex": "PEB(m) = 0.20 * (1/rank) * sqrt(min(streak, 10))",
                "description": "Bonus for consistently top-performing miners",
            },
            {
                "id": "eq5",
                "name": "Final Miner Reward",
                "latex": "R(m) = E_miner * [S_epoch * (1 + PEB)] / sum[S_epoch * (1 + PEB)]",
                "description": "Proportional reward based on epoch score and PEB",
            },
            {
                "id": "eq6",
                "name": "Breakthrough Multiplier",
                "latex": "CMS_breakthrough = CMS * 2.0",
                "description": "2x multiplier for solving previously unsolved tasks with CMS > 0.8",
            },
            {
                "id": "eq7",
                "name": "Validator Accuracy Score (VAS)",
                "latex": "VAS(v) = 1 - (1/|T|) * sum|score_v - score_consensus|",
                "description": "How closely a validator's scores match consensus",
            },
            {
                "id": "eq8",
                "name": "Validator Reward",
                "latex": "R_v = E_val * [VAS * stake * rep] / sum[VAS * stake * rep]",
                "description": "Stake and reputation weighted validator reward",
            },
            {
                "id": "eq9",
                "name": "Trap Penalty",
                "latex": "penalty = max(0, avg_trap / theta_trap) if avg < theta",
                "description": "Penalty for poor performance on known-answer trap tasks",
            },
            {
                "id": "eq10",
                "name": "Validator Slashing",
                "latex": "slash = gamma * stake * (theta_slash - VAS_7d)^2",
                "description": "Quadratic slashing for persistently inaccurate validators",
            },
            {
                "id": "eq11",
                "name": "Objective Score",
                "latex": "O_score = sum(omega_k * check_k)",
                "description": "Domain-specific weighted automated checks",
            },
            {
                "id": "eq12",
                "name": "Consensus Score",
                "latex": "C_score = TrimmedMedian(stake-weighted validator scores)",
                "description": "Stake-weighted trimmed median of validator evaluations",
            },
            {
                "id": "eq13",
                "name": "Final Score",
                "latex": "FinalScore = 0.60 * O_score + 0.40 * C_score",
                "description": "Blend of objective and consensus scores",
            },
        ]
    }
