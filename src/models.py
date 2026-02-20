"""Pydantic v2 data models for Sales Call Copilot."""

from pydantic import BaseModel
from typing import Optional


class CallMetadata(BaseModel):
    """Metadata for an ingested sales call transcript."""

    call_id: str
    filename: str
    date: str  # ISO format string
    title: str


class TranscriptSegment(BaseModel):
    """A single segment/chunk of a sales call transcript."""

    call_id: str
    segment_id: str
    speaker: str = "Unknown"
    timestamp: Optional[str] = None
    text: str
