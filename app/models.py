"""Pydantic request/response models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Models ──


class CreateSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Session name")
    config: Optional[Dict[str, Any]] = Field(None, description="Session-specific settings")


# ── Response Models ──


class SessionSummary(BaseModel):
    id: str
    name: str
    current_stage: str
    pdf_filename: Optional[str] = None
    created_at: str
    updated_at: str
    stats: Optional[Dict[str, Any]] = None


class SessionDetail(BaseModel):
    id: str
    name: str
    current_stage: str
    pdf_filename: Optional[str] = None
    created_at: str
    updated_at: str
    config: Optional[Dict[str, Any]] = None
    checkpoints: List[Dict[str, Any]] = []
    stats: Optional[Dict[str, Any]] = None


class LogEntry(BaseModel):
    id: int
    session_id: str
    stage: Optional[str] = None
    level: str
    message: str
    timestamp: str


class TaskStatus(BaseModel):
    id: str
    session_id: str
    stage: str
    task_type: str
    status: str
    progress_pct: float = 0
    progress_msg: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    success: bool = True
    data: Optional[Any] = None
    error: Optional[ErrorDetail] = None
