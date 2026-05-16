# src/api/models.py  ← NEW FILE
"""
Pydantic models for API request validation and response serialisation.

Every endpoint that receives or returns structured data uses these models.
Pydantic validates all inputs automatically — if a required field is missing
or has the wrong type, FastAPI returns a 422 with a clear error message
before your route function even runs. This is one of FastAPI's biggest
advantages over bare Flask or Django REST.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator


class JobStatus(str, Enum):
    """
    The lifecycle states a job moves through.

    We inherit from both str and Enum so that:
      1. The values serialise to plain strings in JSON (e.g. "PENDING").
      2. You can compare a status to a string literal without .value.
    """
    PENDING           = "PENDING"
    DOWNLOADING       = "DOWNLOADING"
    EXTRACTING_FRAMES = "EXTRACTING_FRAMES"
    DETECTING_FACES   = "DETECTING_FACES"
    RUNNING_INFERENCE = "RUNNING_INFERENCE"
    ANALYZING         = "ANALYZING"
    DONE              = "DONE"
    FAILED            = "FAILED"


class SubmitURLRequest(BaseModel):
    """Request body for POST /api/v1/jobs/url."""
    url: str

    @field_validator("url")
    @classmethod
    def url_must_have_scheme(cls, v: str) -> str:
        # We do a light check here rather than full URL parsing.
        # Heavy URL validation happens inside download_video() where
        # yt-dlp will give us a much more descriptive error message
        # about what is wrong with the specific URL.
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                "URL must begin with http:// or https://. "
                "YouTube links, direct video URLs, and most platform URLs are supported."
            )
        return v.strip()


class JobResponse(BaseModel):
    """Returned by POST (on submission) and GET /jobs/{id} (on poll)."""
    job_id:        str
    status:        JobStatus
    source:        Optional[str]   = None   # the URL or filename submitted
    video_id:      Optional[str]   = None   # set once download completes
    created_at:    float                    # Unix timestamp
    updated_at:    float
    error_message: Optional[str]   = None   # only set when status == FAILED


class ResultResponse(BaseModel):
    """
    Returned by GET /jobs/{id}/result.

    `report` is None while the job is still running.
    This lets the frontend display progress without needing a separate endpoint.
    """
    job_id:        str
    status:        JobStatus
    report:        Optional[Dict[str, Any]] = None
    error_message: Optional[str]            = None