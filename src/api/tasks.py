# src/api/tasks.py  ← NEW FILE
"""
Celery task definitions for the deepfake detection pipeline.

The single task `run_pipeline` executes all five pipeline stages (Days 1–4)
sequentially, updating the job status in SQLite at each transition so the
API can report meaningful progress to the client.

Important design decision: we use ONE task (not a Celery chain of five tasks)
because chaining tasks requires each intermediate result to be serialised
into Redis and deserialised by the next task. Our intermediate results
(VideoMetadata, FrameBatch, etc.) contain Path objects and large lists of
file paths — serialising them safely is complex and adds overhead. Instead,
each stage writes its output to disk (manifest.json, face_manifest.json, etc.)
and the next stage reads from disk. The disk IS the message between stages.
This is also more resilient: if the worker restarts mid-pipeline, the completed
stages' outputs are still on disk and can be inspected manually.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.api.celery_app import celery_app
from src.api.database import update_job
from src.api.models import JobStatus
from src.utils.logger import logger


@celery_app.task(
    bind=True,
    name="tasks.run_pipeline",
    max_retries=0,          # do not auto-retry — failed jobs need human review
    soft_time_limit=1800,   # 30-minute soft limit — raises SoftTimeLimitExceeded
    time_limit=1900,        # 31-minute hard limit — kills the worker process
)
def run_pipeline(self, job_id: str, source: str) -> dict:
    """
    Execute the full deepfake detection pipeline for one video.

    Args:
        job_id:  UUID string identifying the job in the SQLite database.
        source:  Either a URL string (for web submissions) or an absolute
                 file path string (for uploaded files).

    Returns:
        A plain dict with the final verdict and video_id, stored in Redis
        by Celery as the task result. We also write final_report.json to disk.

    The `bind=True` argument gives us access to `self` — the task instance —
    which we use to log the Celery task ID alongside our job_id for debugging.
    """
    log = logger.bind(job_id=job_id, celery_task_id=self.request.id)
    log.info("pipeline_started", source=source)

    try:
        # ── Stage 1: Download & validate ──────────────────────────────────
        # Import here (inside the task) rather than at module top-level.
        # Celery workers import tasks.py at startup. Top-level imports of
        # heavy ML libraries (torch, onnxruntime) would slow worker startup
        # significantly. Importing inside the function body defers that cost
        # to when the task actually runs.
        update_job(job_id, JobStatus.DOWNLOADING)
        log.info("stage_download_started")

        from config import settings
        from src.ingestion.downloader import download_video

        meta = download_video(source)
        log.info(
            "stage_download_complete",
            video_id=meta.video_id,
            duration=meta.duration_seconds,
        )

        # ── Stage 2: Frame extraction ──────────────────────────────────────
        update_job(job_id, JobStatus.EXTRACTING_FRAMES, video_id=meta.video_id)
        log.info("stage_frames_started")

        from src.ingestion.frame_extractor import extract_frames

        frame_batch = extract_frames(
            video_path=meta.local_path,
            video_id=meta.video_id,
            duration_seconds=meta.duration_seconds,
            fps=meta.fps,
            strategy="hybrid",
        )

        # extract_frames returns a FrameBatch but does NOT write manifest.json.
        # The CLI wrote it manually; we do the same here so that
        # extract_faces_from_manifest (Stage 3) can read it.
        manifest_path = settings.raw_frames_dir / meta.video_id / "manifest.json"
        manifest_data = {
            "video_id":    meta.video_id,
            "source":      meta.source,
            "frames_dir":  str(frame_batch.frames_dir),
            "frame_paths": [str(p) for p in frame_batch.frame_paths],
            "frame_count": frame_batch.count,
            "strategy":    frame_batch.extraction_strategy,
            "video":       meta.to_dict(),
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        log.info("stage_frames_complete", frame_count=frame_batch.count)

        # ── Stage 3: Face detection & alignment ───────────────────────────
        update_job(job_id, JobStatus.DETECTING_FACES)
        log.info("stage_faces_started")

        from src.detection.face_extractor import extract_faces_from_manifest

        face_batch = extract_faces_from_manifest(manifest_path=manifest_path)
        log.info("stage_faces_complete", face_count=face_batch.count)

        # ── Stage 4: Model inference ──────────────────────────────────────
        update_job(job_id, JobStatus.RUNNING_INFERENCE)
        log.info("stage_inference_started")

        from src.detection.video_scorer import score_video_from_manifest

        face_manifest_path = (
            settings.face_crops_dir / meta.video_id / "face_manifest.json"
        )
        inference_result = score_video_from_manifest(
            face_manifest_path=face_manifest_path
        )
        log.info(
            "stage_inference_complete",
            verdict=inference_result.verdict,
            faces_scored=inference_result.total_faces_scored,
        )

        # ── Stage 5: Temporal analysis, Grad-CAM, confidence intervals ────
        update_job(job_id, JobStatus.ANALYZING)
        log.info("stage_analysis_started")

        from src.scoring.report_builder import build_report

        inference_result_path = (
            settings.face_crops_dir / meta.video_id / "inference_result.json"
        )
        report = build_report(inference_result_path=inference_result_path)
        log.info("stage_analysis_complete", verdict=report.verdict)

        # ── Mark job as DONE ──────────────────────────────────────────────
        result_path = settings.results_dir / meta.video_id / "final_report.json"
        update_job(
            job_id,
            JobStatus.DONE,
            result_path=str(result_path),
        )

        log.info(
            "pipeline_complete",
            verdict=report.verdict,
            result_path=str(result_path),
        )

        return {
            "verdict":  report.verdict,
            "video_id": meta.video_id,
        }

    except Exception as exc:
        # Catch everything — log it, mark the job FAILED, then re-raise
        # so Celery records the failure in Redis too. This dual logging
        # means you can find the failure in either place.
        error_msg = str(exc)
        log.error("pipeline_failed", error=error_msg, exc_info=True)
        update_job(job_id, JobStatus.FAILED, error_message=error_msg[:2000])
        raise