"""HTTP API request/response models."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the video file as the server sees it")
    target_languages: List[str] = Field(..., min_length=1)
    source_language: Optional[str] = Field(None, description="ISO 639-1 code; null = auto-detect")
    bilingual: bool = False
    keep_original: bool = True


class JobOutput(BaseModel):
    language: str
    path: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    video_path: str
    target_languages: List[str]
    source_language: Optional[str] = None
    bilingual: bool = False
    keep_original: bool = True
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    outputs: List[JobOutput] = []


class JobAccepted(BaseModel):
    job_id: str
    status: str = "pending"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    queue_pending: int
    queue_processing: int
    transcriber_loaded: bool
