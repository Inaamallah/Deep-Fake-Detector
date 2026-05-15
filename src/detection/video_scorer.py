# src/detection/video_scorer.py
"""
Aggregates per-face scores into a video-level verdict.

Why not just take a simple mean?
---------------------------------
Consider a 5-minute video with 100 frames. MediaPipe finds faces in 90 of
them. Of those, 10 are motion-blurred (low quality, low-confidence scores
near 0.5). If we average naively, those 10 uncertain frames dilute the
signal from the 80 clean frames. Confidence-weighting solves this by
giving near-zero weight to frames where the model is unsure.

The formula for each face score s_i with confidence c_i is:
    weighted_avg = Σ(s_i * w_i) / Σ(w_i)
where the weight w_i is defined as:
    w_i = |s_i - 0.5| * 2      (how far the score is from the decision boundary)
A score of 0.9 → weight 0.8 (strong signal)
A score of 0.52 → weight 0.04 (nearly useless — near the boundary)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import settings
from src.detection.inference_engine import (
    DeepfakeInferenceEngine,
    FaceScore,
    VideoInferenceResult,
)
from src.utils.logger import logger


def _confidence_weights(probs: np.ndarray) -> np.ndarray:
    """
    Weight = distance from the decision boundary (0.5), scaled to [0, 1].
    Scores close to 0.5 (uncertain) get near-zero weight.
    Scores close to 0.0 or 1.0 (decisive) get weight close to 1.0.
    """
    weights = np.abs(probs - 0.5) * 2.0
    # Add a small epsilon to avoid division by zero when all scores = 0.5
    return weights + 1e-6


def aggregate_scores(scores: List[FaceScore], threshold: float = 0.5) -> dict:
    """
    Compute video-level statistics from a list of per-face scores.

    Returns a dict with mean_prob_fake, weighted_prob_fake, verdict, confidence.
    """
    if not scores:
        return {
            "mean_prob_fake": 0.5,
            "weighted_prob_fake": 0.5,
            "verdict": "INCONCLUSIVE",
            "overall_confidence": 0.0,
            "fake_frame_ratio": 0.0,
        }

    probs = np.array([s.prob_fake for s in scores], dtype=np.float32)
    weights = _confidence_weights(probs)

    mean_prob      = float(np.mean(probs))
    weighted_prob  = float(np.average(probs, weights=weights))
    fake_ratio     = float(np.mean(probs >= threshold))

    verdict = "DEEPFAKE" if weighted_prob >= threshold else "REAL"
    if verdict == "DEEPFAKE":
        confidence_scale = max(1.0 - threshold, 1e-6)
        overall_confidence = (weighted_prob - threshold) / confidence_scale
    else:
        confidence_scale = max(threshold, 1e-6)
        overall_confidence = (threshold - weighted_prob) / confidence_scale
    overall_confidence = float(np.clip(overall_confidence, 0.0, 1.0))

    return {
        "mean_prob_fake":    round(mean_prob, 4),
        "weighted_prob_fake": round(weighted_prob, 4),
        "verdict":           verdict,
        "overall_confidence": round(overall_confidence, 4),
        "fake_frame_ratio":  round(fake_ratio, 4),
        "decision_threshold": round(float(threshold), 4),
    }


def score_video_from_manifest(
    face_manifest_path: Path,
    engine: Optional[DeepfakeInferenceEngine] = None,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> VideoInferenceResult:
    """
    Full Day 3 pipeline: read face_manifest → run inference → aggregate → save.

    Args:
        face_manifest_path: Path to face_manifest.json written by Day 2.
        engine:             Optionally pass a pre-built engine (for reuse
                            across multiple videos in a queue).
        batch_size:         Inference batch size. Higher = more RAM, more speed.

    Returns:
        VideoInferenceResult with all scores and the final verdict.
    """
    log = logger.bind(component="video_scorer")
    manifest = json.loads(face_manifest_path.read_text())
    video_id = manifest["video_id"]
    records  = manifest["records"]

    log = log.bind(video_id=video_id, face_count=len(records))
    log.info("scoring_starting")

    if not records:
        log.warning("no_face_records_in_manifest")
        return VideoInferenceResult(
            video_id=video_id,
            face_scores=[],
            total_faces_scored=0,
            mean_prob_fake=0.5,
            weighted_prob_fake=0.5,
            verdict="INCONCLUSIVE",
            overall_confidence=0.0,
            fake_frame_ratio=0.0,
            decision_threshold=threshold,
            elapsed_seconds=0.0,
        )

    # Build engine if not provided (happens on first call)
    if engine is None:
        engine = DeepfakeInferenceEngine(batch_size=batch_size)

    start = time.time()

    # Unpack the manifest into typed lists
    crop_paths    = [Path(r["crop_path"])  for r in records]
    frame_indices = [r["frame_idx"]        for r in records]
    face_indices  = [r["face_idx"]         for r in records]

    # Run inference
    face_scores = engine.score_crop_list(
        crop_paths,
        frame_indices,
        face_indices,
        threshold=threshold,
    )

    # Aggregate
    agg = aggregate_scores(face_scores, threshold=threshold)
    elapsed = time.time() - start

    result = VideoInferenceResult(
        video_id=video_id,
        face_scores=face_scores,
        total_faces_scored=len(face_scores),
        elapsed_seconds=round(elapsed, 2),
        **agg,
    )

    log.info(
        "scoring_complete",
        verdict=result.verdict,
        weighted_prob=result.weighted_prob_fake,
        confidence=result.overall_confidence,
        elapsed_s=elapsed,
    )

    # Persist result for Day 4 to pick up
    _save_result(result, face_manifest_path.parent)
    return result


def _save_result(result: VideoInferenceResult, output_dir: Path) -> None:
    """Write the inference result as JSON alongside the face crops."""
    out = {
        "video_id":           result.video_id,
        "verdict":            result.verdict,
        "weighted_prob_fake": result.weighted_prob_fake,
        "mean_prob_fake":     result.mean_prob_fake,
        "overall_confidence": result.overall_confidence,
        "fake_frame_ratio":   result.fake_frame_ratio
        if hasattr(result, "fake_frame_ratio") else None,
        "decision_threshold": result.decision_threshold,
        "total_faces_scored": result.total_faces_scored,
        "elapsed_seconds":    result.elapsed_seconds,
        "face_scores": [
            {
                "frame_idx":  s.frame_idx,
                "face_idx":   s.face_idx,
                "prob_fake":  s.prob_fake,
                "is_fake":    s.is_fake,
                "crop_path":  s.crop_path,
                "latency_ms": s.latency_ms,
            }
            for s in result.face_scores
        ],
    }
    result_path = output_dir / "inference_result.json"
    result_path.write_text(json.dumps(out, indent=2))
    logger.info("inference_result_saved", path=str(result_path))
