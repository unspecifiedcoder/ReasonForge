"""
ReasonForge - Gateway Request/Response Schemas

Pydantic models for the external API gateway.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskSubmissionRequest(BaseModel):
    """Request to submit a reasoning task to the network."""
    problem: str = Field(..., min_length=10, max_length=10000)
    domain: Optional[str] = None
    difficulty: Optional[int] = Field(None, ge=1, le=10)
    timeout_seconds: Optional[int] = Field(300, ge=30, le=600)
    callback_url: Optional[str] = None


class TaskResultResponse(BaseModel):
    """Response for task result queries."""
    task_id: str
    status: str  # "queued" | "processing" | "completed" | "failed"
    result: Optional[Dict] = None
    best_answer: Optional[str] = None
    confidence: Optional[float] = None
    reasoning_steps: Optional[List[Dict]] = None
    processing_time_ms: Optional[int] = None


class LeaderboardEntry(BaseModel):
    """Single entry in the miner leaderboard."""
    uid: int
    s_epoch: float
    peb: float
    rank: int
    streak: int
    tasks_completed: int


class LeaderboardResponse(BaseModel):
    """Miner leaderboard response."""
    epoch_id: int
    entries: List[LeaderboardEntry]
    total_miners: int


class NetworkStatsResponse(BaseModel):
    """Network statistics response."""
    current_epoch: int
    total_tasks_processed: int
    active_miners: int
    active_validators: int
    avg_cms: float
    total_emission_tao: float
    top_domains: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    epoch: int
    db_connected: bool


class APIKeyInfo(BaseModel):
    """API key information."""
    key_id: str
    owner: str
    tier: str
    requests_used: int
    requests_limit: int
