# src/api/routes.py  ← NEW FILE
"""
FastAPI route handlers for the deepfake detection API.

All routes are registered on a single APIRouter with prefix /api/v1.
This router is mounted onto the main FastAPI app in app.py, which keeps
route definitions separate from application-level configuration (middleware,
lifespan events, exception handlers).
"""
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from config import settings
from src.api.auth import verify_api_key
from src.api.database import create_job, get_job
from src.api.limiter import limiter
from src.api.models import JobResponse, JobStatus, ResultResponse, SubmitURLRequest
from src.api.tasks import run_pipeline
from src.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["Deepfake Detection"])

# Directory where uploaded video files are temporarily stored
# before being handed off to the Celery worker.
_UPLOAD_DIR = settings.data_dir / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def _row_to_response(job: dict) -> JobResponse:
    """Convert a raw SQLite row dict to a typed JobResponse model."""
    return JobResponse(
        job_id        = job["job_id"],
        status        = JobStatus(job["status"]),
        source        = job.get("source"),
        video_id      = job.get("video_id"),
        created_at    = job["created_at"],
        updated_at    = job["updated_at"],
        error_message = job.get("error_message"),
    )


# ---------------------------------------------------------------------------
# Health check — no auth required
# ---------------------------------------------------------------------------

@router.get("/health", tags=["Health"])
async def health_check():
    """
    Simple liveness probe. Returns 200 if the API is up.
    No auth required so load balancers and monitoring tools can use it freely.
    """
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Job submission — two variants: URL and file upload
# ---------------------------------------------------------------------------

@router.post(
    "/jobs/url",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a video URL for deepfake analysis",
    description=(
        "Accepts any yt-dlp-compatible URL (YouTube, Vimeo, TikTok, etc.) "
        "or a direct .mp4/.webm link. Returns a job_id immediately. "
        "Poll GET /jobs/{job_id} to track progress."
    ),
)
@limiter.limit("10/minute")   # stricter limit for URL submissions
async def submit_url(
    request:  Request,                          # required by slowapi rate limiter
    body:     SubmitURLRequest,
    _api_key: str = Depends(verify_api_key),
) -> JobResponse:
    """Submit a URL and get a job_id back in under 100ms."""
    job_id = create_job(source=body.url)

    # .delay() is Celery's shorthand for .apply_async() with no special options.
    # It serialises the arguments to JSON, pushes them to Redis, and returns
    # an AsyncResult object — which we deliberately ignore here because we
    # track state in SQLite, not in Celery's result backend.
    run_pipeline.delay(job_id=job_id, source=body.url)

    logger.info("job_created_url", job_id=job_id, url=body.url)
    job = get_job(job_id)
    return _row_to_response(job)


@router.post(
    "/jobs/upload",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a video file for deepfake analysis",
    description=(
        "Accepts mp4, mov, avi, webm, mkv, m4v. "
        f"Maximum file size: {settings.max_video_size_mb}MB. "
        "The file is saved to disk before the job is queued, so the response "
        "may take a few seconds for large files."
    ),
)
@limiter.limit("5/minute")    # lower limit — uploads are heavier than URL submissions
async def submit_upload(
    request:  Request,
    file:     UploadFile = File(..., description="Video file to analyse"),
    _api_key: str = Depends(verify_api_key),
) -> JobResponse:
    """Save an uploaded file and queue it for analysis."""

    # Validate the file extension before reading any bytes
    original_name = file.filename or "video.mp4"
    suffix = Path(original_name).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Allowed types: {', '.join(sorted(_ALLOWED_EXTENSIONS))}."
            ),
        )

    # Read the entire file into memory to check its size before saving.
    # This is safe because our max_video_size_mb limit (default 500MB) is
    # enforced here before writing to disk. For truly large file support
    # you would stream in chunks, but that complicates size enforcement.
    content = await file.read()
    size_mb = len(content) / 1_000_000

    if size_mb > settings.max_video_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {size_mb:.1f}MB — exceeds the "
                f"{settings.max_video_size_mb}MB limit."
            ),
        )

    # Use a UUID prefix to guarantee uniqueness even if two users
    # upload files with the same name concurrently.
    unique_name = f"{uuid.uuid4()}{suffix}"
    dest_path   = _UPLOAD_DIR / unique_name

    try:
        dest_path.write_bytes(content)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    source = str(dest_path)
    job_id = create_job(source=source)
    run_pipeline.delay(job_id=job_id, source=source)

    logger.info(
        "job_created_upload",
        job_id=job_id,
        filename=original_name,
        size_mb=round(size_mb, 2),
    )
    job = get_job(job_id)
    return _row_to_response(job)


# ---------------------------------------------------------------------------
# Job status polling
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Poll the status of a submitted job",
)
@limiter.limit("60/minute")   # higher limit — clients poll this frequently
async def get_job_status(
    request:  Request,
    job_id:   str,
    _api_key: str = Depends(verify_api_key),
) -> JobResponse:
    """
    Returns the current status of a job. Clients should poll this every
    3–5 seconds until status is DONE or FAILED.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return _row_to_response(job)


# ---------------------------------------------------------------------------
# Result retrieval
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}/result",
    response_model=ResultResponse,
    summary="Get the final analysis report for a completed job",
)
async def get_result(
    job_id:   str,
    _api_key: str = Depends(verify_api_key),
) -> ResultResponse:
    """
    Returns the complete final_report.json content when a job is DONE.
    Returns the report field as null with the current status for jobs still running.
    Returns the error message for FAILED jobs.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    job_status = JobStatus(job["status"])

    # Still running — return status with no report (null).
    # The client can decide whether to keep polling.
    if job_status not in (JobStatus.DONE, JobStatus.FAILED):
        return ResultResponse(job_id=job_id, status=job_status)

    # Failed — return the error message so the client can show it.
    if job_status == JobStatus.FAILED:
        return ResultResponse(
            job_id=job_id,
            status=job_status,
            error_message=job.get("error_message"),
        )

    # Done — load and return the report.
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Job is marked DONE but the result file is missing. "
                "This indicates a server-side storage problem."
            ),
        )

    import json
    report = json.loads(Path(result_path).read_text())

    return ResultResponse(job_id=job_id, status=job_status, report=report)


# ---------------------------------------------------------------------------
# Heatmap image serving
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}/heatmaps/{filename}",
    summary="Serve a Grad-CAM heatmap image",
    response_class=FileResponse,
)
async def get_heatmap(
    job_id:   str,
    filename: str,
    _api_key: str = Depends(verify_api_key),
):
    """
    Serves PNG heatmap images generated during the Grad-CAM analysis step.
    Filename must match one of the entries in final_report.json's heatmaps list.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    if JobStatus(job["status"]) != JobStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Heatmaps are only available after the job completes.",
        )

    video_id = job.get("video_id")
    if not video_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job has no associated video_id.",
        )

    # Path traversal prevention: Path(filename).name strips any directory
    # components, so "../../../etc/passwd" becomes "passwd" — harmless.
    safe_name    = Path(filename).name
    heatmap_path = settings.results_dir / video_id / "heatmaps" / safe_name

    if not heatmap_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Heatmap '{safe_name}' not found for this job.",
        )

    return FileResponse(str(heatmap_path), media_type="image/png")