"""
ReasonForge - API Gateway Application

External-facing FastAPI application for users to submit tasks and query results.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .auth import APIKeyManager
from .billing import BillingTracker
from .rate_limiter import PerIPRateLimiter
from .schemas import (
    HealthResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    NetworkStatsResponse,
    TaskResultResponse,
    TaskSubmissionRequest,
)

# ── App Setup ──

app = FastAPI(
    title="ReasonForge Gateway",
    version="0.1.0",
    description="External API for the ReasonForge Bittensor Subnet",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State (initialized on startup) ──

_start_time = time.time()
_auth_manager = APIKeyManager()
_billing = BillingTracker()
_rate_limiter = PerIPRateLimiter(requests_per_minute=60)

# Task queue (in production, this would be shared with the validator)
_task_queue: dict = {}
_task_results: dict = {}
_epoch_data: dict = {"epoch_id": 0, "miner_states": {}}


# ── Auth Dependency ──

async def verify_api_key(x_api_key: str = Header(None)) -> dict:
    """Verify API key from request header."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    info = _auth_manager.verify_key(x_api_key)
    if not info:
        raise HTTPException(status_code=403, detail="Invalid or expired API key")
    return {"key": x_api_key, "info": info}


# ── Rate Limit Dependency ──

async def check_rate_limit(request: Request) -> None:
    """Check per-IP rate limits."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )


# ── Endpoints ──

@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=time.time() - _start_time,
        epoch=_epoch_data.get("epoch_id", 0),
        db_connected=True,
    )


@app.post(
    "/v1/tasks",
    response_model=TaskResultResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def submit_task(
    request: TaskSubmissionRequest,
    auth: dict = Depends(verify_api_key),
):
    """Submit a reasoning task to the network."""
    task_id = str(uuid.uuid4())

    _task_queue[task_id] = {
        "task_id": task_id,
        "problem": request.problem,
        "domain": request.domain,
        "difficulty": request.difficulty,
        "timeout_seconds": request.timeout_seconds,
        "callback_url": request.callback_url,
        "status": "queued",
        "submitted_at": time.time(),
    }

    # Track billing
    _billing.record_usage(
        key_id=auth["info"].key_id,
        task_id=task_id,
        domain=request.domain or "auto",
    )
    _auth_manager.track_usage(auth["key"])

    return TaskResultResponse(task_id=task_id, status="queued")


@app.get(
    "/v1/tasks/{task_id}",
    response_model=TaskResultResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def get_task_result(task_id: str):
    """Poll for task results."""
    # Check task queue
    if task_id in _task_queue:
        task = _task_queue[task_id]
        return TaskResultResponse(
            task_id=task_id,
            status=task["status"],
        )

    # Check completed results
    if task_id in _task_results:
        result = _task_results[task_id]
        return TaskResultResponse(
            task_id=task_id,
            status="completed",
            result=result.get("result"),
            best_answer=result.get("best_answer"),
            confidence=result.get("confidence"),
            reasoning_steps=result.get("reasoning_steps"),
            processing_time_ms=result.get("processing_time_ms"),
        )

    raise HTTPException(status_code=404, detail="Task not found")


@app.get(
    "/v1/leaderboard",
    response_model=LeaderboardResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def get_leaderboard(domain: Optional[str] = None, limit: int = 20):
    """Get current miner rankings."""
    miners = _epoch_data.get("miner_states", {})
    entries = []

    for uid_str, ms in miners.items():
        entries.append(LeaderboardEntry(
            uid=int(uid_str),
            s_epoch=ms.get("s_epoch", 0.0),
            peb=ms.get("peb", 0.0),
            rank=ms.get("rank", 0),
            streak=ms.get("streak", 0),
            tasks_completed=ms.get("task_count", 0),
        ))

    entries.sort(key=lambda e: e.s_epoch, reverse=True)
    entries = entries[:limit]

    return LeaderboardResponse(
        epoch_id=_epoch_data.get("epoch_id", 0),
        entries=entries,
        total_miners=len(miners),
    )


@app.get(
    "/v1/stats",
    response_model=NetworkStatsResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def get_stats():
    """Get network statistics."""
    miners = _epoch_data.get("miner_states", {})
    avg_cms = 0.0
    if miners:
        scores = [ms.get("s_epoch", 0) for ms in miners.values()]
        avg_cms = sum(scores) / len(scores) if scores else 0.0

    return NetworkStatsResponse(
        current_epoch=_epoch_data.get("epoch_id", 0),
        total_tasks_processed=len(_task_results),
        active_miners=len(miners),
        active_validators=1,
        avg_cms=avg_cms,
        total_emission_tao=0.0,
        top_domains={},
    )


def create_app(db=None, auth_manager=None) -> FastAPI:
    """Factory for creating the gateway app with dependencies."""
    global _auth_manager, _billing

    if auth_manager:
        _auth_manager = auth_manager
    if db:
        _auth_manager = APIKeyManager(db=db)
        _billing = BillingTracker(db=db)

    return app
