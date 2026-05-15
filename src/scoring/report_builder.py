# src/scoring/report_builder.py  ← NEW FILE
"""
Assembles the complete Day 4 analysis report from all component outputs.

Reads:   data/face_crops/<video_id>/inference_result.json   (Day 3)
Writes:  data/results/<video_id>/final_report.json          (Day 4)
         data/results/<video_id>/heatmaps/*.png             (Day 4)

The final_report.json is the canonical output of the entire pipeline.
Days 5–7 (API, frontend, Docker) read this file — nothing else.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import settings
from src.scoring.confidence_estimator import ConfidenceEstimate, estimate_confidence
from src.scoring.gradcam import GradCAMGenerator, HeatmapResult
from src.scoring.temporal_analyzer import TemporalAnalysis, analyze_temporal
from src.utils.logger import logger


@dataclass
class FinalReport:
    """
    The complete analysis output for one video.
    All downstream components (API, frontend) read from this structure.
    """
    video_id:            str
    verdict:             str          # "REAL" | "DEEPFAKE" | "INCONCLUSIVE"
    weighted_prob_fake:  float        # confidence-weighted mean P(fake)
    mean_prob_fake:      float        # simple mean P(fake)
    overall_confidence:  float        # |weighted_prob - 0.5| * 2, from Day 3
    confidence_interval: dict         # from ConfidenceEstimate
    temporal:            dict         # from TemporalAnalysis.to_dict()
    top_suspicious_frames: List[dict] # highest-scoring frames, sorted desc
    heatmaps:            List[dict]   # HeatmapResult metadata
    total_faces_scored:  int
    generated_at:        str          # ISO-8601 UTC timestamp
    elapsed_seconds:     float

    def to_dict(self) -> dict:
        return asdict(self)


def _top_n_frames(face_scores: List[dict], n: int) -> List[dict]:
    """Return the n highest P(fake) face score records, sorted descending."""
    return sorted(face_scores, key=lambda s: s["prob_fake"], reverse=True)[:n]


def build_report(
    inference_result_path: Path,
    top_n_heatmaps:  int = 5,
    smooth_window:   int = 5,
    n_bootstrap:     int = 2000,
) -> FinalReport:
    """
    Full Day 4 pipeline for one video.

    Args:
        inference_result_path: Path to inference_result.json from Day 3.
                               Expected location:
                               data/face_crops/<video_id>/inference_result.json
        top_n_heatmaps:        Number of most-suspicious frames to generate
                               Grad-CAM heatmaps for. 5 is a reasonable default
                               — more than 10 gives diminishing analytical value.
        smooth_window:         Gaussian window size for temporal smoothing.
        n_bootstrap:           Bootstrap iterations for confidence intervals.

    Returns:
        FinalReport dataclass. Also writes final_report.json to disk.
    """
    log   = logger.bind(component="report_builder")
    start = time.time()

    # ── Load Day 3 output ──────────────────────────────────────────────────
    raw          = json.loads(inference_result_path.read_text())
    video_id     = raw["video_id"]
    verdict      = raw["verdict"]
    face_scores  = raw.get("face_scores", [])

    log = log.bind(video_id=video_id, face_count=len(face_scores))
    log.info("report_build_starting")

    # ── Prepare output directories ─────────────────────────────────────────
    results_dir  = settings.results_dir / video_id
    heatmaps_dir = results_dir / "heatmaps"
    results_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)

    # ── Temporal analysis ──────────────────────────────────────────────────
    log.info("running_temporal_analysis")

    frame_indices = [s["frame_idx"] for s in face_scores]
    prob_fakes    = [s["prob_fake"]  for s in face_scores]

    temporal: TemporalAnalysis = analyze_temporal(
        frame_indices      = frame_indices,
        prob_fakes         = prob_fakes,
        smooth_window      = smooth_window,
    )

    log.info(
        "temporal_analysis_complete",
        temporal_verdict      = temporal.temporal_verdict,
        suspicious_windows    = len(temporal.suspicious_windows),
        suspicious_frame_ratio = temporal.suspicious_frame_ratio,
        run_length_score      = temporal.run_length_score,
    )

    # ── Confidence estimation ──────────────────────────────────────────────
    log.info("running_confidence_estimation", n_bootstrap=n_bootstrap)

    confidence: ConfidenceEstimate = estimate_confidence(
        prob_fakes  = prob_fakes,
        verdict     = verdict,
        n_bootstrap = n_bootstrap,
    )

    log.info(
        "confidence_estimation_complete",
        ci_95        = f"[{confidence.ci_lower_95:.3f}, {confidence.ci_upper_95:.3f}]",
        bootstrap_std = confidence.bootstrap_std,
    )

    # ── Grad-CAM heatmaps ──────────────────────────────────────────────────
    top_frames     = _top_n_frames(face_scores, n=top_n_heatmaps)
    heatmap_results: List[HeatmapResult] = []

    if top_frames:
        log.info("generating_heatmaps", n=len(top_frames))
        try:
            generator = GradCAMGenerator(output_dir=heatmaps_dir)
            heatmap_results = generator.generate_for_crops(
                crop_paths    = [f["crop_path"]  for f in top_frames],
                frame_indices = [f["frame_idx"]  for f in top_frames],
                prob_fakes    = [f["prob_fake"]   for f in top_frames],
            )
            generator.close()
            log.info("heatmaps_complete", count=len(heatmap_results))
        except Exception as exc:
            # Grad-CAM failure is non-fatal — the rest of the report is still valid.
            log.warning(
                "heatmap_generation_failed",
                error=str(exc),
                note="Report will be saved without heatmaps.",
            )

    # ── Assemble FinalReport ───────────────────────────────────────────────
    elapsed = time.time() - start

    report = FinalReport(
        video_id            = video_id,
        verdict             = verdict,
        weighted_prob_fake  = raw.get("weighted_prob_fake", 0.5),
        mean_prob_fake      = raw.get("mean_prob_fake", 0.5),
        overall_confidence  = raw.get("overall_confidence", 0.0),
        confidence_interval = {
            "point_estimate": confidence.point_estimate,
            "ci_lower_95":    confidence.ci_lower_95,
            "ci_upper_95":    confidence.ci_upper_95,
            "ci_lower_90":    confidence.ci_lower_90,
            "ci_upper_90":    confidence.ci_upper_90,
            "bootstrap_std":  confidence.bootstrap_std,
            "n_samples":      confidence.n_samples_used,
            "interpretation": confidence.interpretation,
        },
        temporal            = temporal.to_dict(),
        top_suspicious_frames = top_frames,
        heatmaps = [
            {
                "frame_idx":    h.frame_idx,
                "prob_fake":    h.prob_fake,
                "crop_path":    h.crop_path,
                "heatmap_path": h.heatmap_path,
                "overlay_path": h.overlay_path,
            }
            for h in heatmap_results
        ],
        total_faces_scored  = raw.get("total_faces_scored", len(face_scores)),
        generated_at        = datetime.now(timezone.utc).isoformat(),
        elapsed_seconds     = round(elapsed, 2),
    )

    # ── Save to disk ───────────────────────────────────────────────────────
    report_path = results_dir / "final_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    log.info(
        "report_saved",
        path=str(report_path),
        elapsed_s=round(elapsed, 2),
    )

    return report