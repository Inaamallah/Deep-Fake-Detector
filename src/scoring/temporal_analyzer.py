# src/scoring/temporal_analyzer.py  ← NEW FILE
"""
Temporal analysis of per-frame deepfake scores.

The core insight: a single high-scoring frame could be noise —
a blurry frame, unusual lighting, a fast head turn.
But a *cluster* of consecutive high-scoring frames is a meaningful
signal that a specific segment of the video was manipulated.

This module analyses the time series of frame scores to detect:
  1. Suspicious windows — contiguous runs of high scores
  2. Run-length clustering — are fake scores grouped or scattered?
  3. Temporal consistency — does the model's certainty hold over time?
  4. A temporal verdict label that summarises the pattern type.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np


@dataclass
class SuspiciousWindow:
    """A contiguous segment of frames where smoothed P(fake) exceeds the threshold."""
    start_frame: int
    end_frame: int
    mean_prob_fake: float
    max_prob_fake: float
    frame_count: int


@dataclass
class TemporalAnalysis:
    """Complete temporal analysis result for one video."""
    frame_indices: List[int]
    raw_scores: List[float]
    smoothed_scores: List[float]
    suspicious_windows: List[SuspiciousWindow]
    score_variance: float
    peak_frame_idx: int
    peak_score: float
    suspicious_frame_ratio: float
    run_length_score: float
    temporal_verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_smooth(scores: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply Gaussian-weighted smoothing over the score time series.

    Why Gaussian rather than a simple moving average?
    A simple average weights every frame in the window equally — a spike
    at the edge of the window has the same influence as one at the centre.
    Gaussian weighting tapers off toward the edges, so the smoothed value
    at each frame is dominated by its nearest temporal neighbours.
    This better preserves genuine score transitions while suppressing noise.

    Edge handling: we pad with edge values (not zeros) so the smoothed
    curve doesn't artificially dip toward zero at the start/end of the video.
    """
    if len(scores) < 2:
        return scores.copy()

    half = window_size // 2
    sigma = window_size / 3.0
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    padded = np.pad(scores, half, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    # convolve with mode="valid" may produce slightly different length
    # depending on padding — trim to original length defensively.
    return smoothed[: len(scores)].astype(np.float32)


def _find_suspicious_windows(
    frame_indices: List[int],
    smoothed: np.ndarray,
    threshold: float,
    min_length: int,
) -> List[SuspiciousWindow]:
    """
    Identify contiguous runs of frames where the smoothed score >= threshold.

    Only windows with at least `min_length` frames are kept — isolated
    single-frame spikes are almost always noise and are discarded.

    The state machine tracks whether we are currently inside a suspicious
    window and closes/opens windows as scores cross the threshold.
    """
    windows: List[SuspiciousWindow] = []
    in_window = False
    win_start_pos = 0
    win_scores: List[float] = []

    for pos, (fidx, score) in enumerate(zip(frame_indices, smoothed)):
        above = float(score) >= threshold

        if above and not in_window:
            in_window = True
            win_start_pos = pos
            win_scores = [float(score)]

        elif above and in_window:
            win_scores.append(float(score))

        elif not above and in_window:
            in_window = False
            if len(win_scores) >= min_length:
                windows.append(SuspiciousWindow(
                    start_frame=frame_indices[win_start_pos],
                    end_frame=frame_indices[pos - 1],
                    mean_prob_fake=round(float(np.mean(win_scores)), 4),
                    max_prob_fake=round(float(np.max(win_scores)), 4),
                    frame_count=len(win_scores),
                ))
            win_scores = []

    # Close any window that reaches the end of the video
    if in_window and len(win_scores) >= min_length:
        windows.append(SuspiciousWindow(
            start_frame=frame_indices[win_start_pos],
            end_frame=frame_indices[-1],
            mean_prob_fake=round(float(np.mean(win_scores)), 4),
            max_prob_fake=round(float(np.max(win_scores)), 4),
            frame_count=len(win_scores),
        ))

    return windows


def _compute_run_length_score(
    scores: np.ndarray, threshold: float
) -> float:
    """
    Measure whether fake-scoring frames cluster together more than chance.

    For a random binary sequence where P(fake) = p, the expected mean run
    length of consecutive fake frames is 1 / (1 - p).

    We compute the *actual* mean run length from the data and divide by
    this expectation:
        run_length_score = actual_mean / expected_mean

    Score > 1  → fake frames cluster together (suspicious, typical of
                 segment-level manipulation)
    Score ≈ 1  → fake frames are randomly distributed (less systematic)
    Score < 1  → fake frames are more isolated than chance

    Returns 1.0 for degenerate inputs (all real or all fake).
    """
    binary = (scores >= threshold).astype(np.int32)
    p_fake = float(binary.mean())

    if p_fake <= 0.0 or p_fake >= 1.0:
        return 1.0

    # Collect run lengths of consecutive 1s
    run_lengths: List[int] = []
    current = 0
    for b in binary:
        if b == 1:
            current += 1
        elif current > 0:
            run_lengths.append(current)
            current = 0
    if current > 0:
        run_lengths.append(current)

    if not run_lengths:
        return 1.0

    actual_mean   = float(np.mean(run_lengths))
    expected_mean = 1.0 / max(1.0 - p_fake, 1e-8)

    return round(float(actual_mean / expected_mean), 4)


def _classify_temporal_pattern(
    smoothed: np.ndarray,
    suspicious_windows: List[SuspiciousWindow],
    suspicious_frame_ratio: float,
) -> str:
    """
    Map temporal statistics to one of four human-readable verdict labels.

    CONSISTENT_FAKE  – High mean, low variance: the whole video is likely fake.
    CONSISTENT_REAL  – Low mean, low variance: the whole video appears authentic.
    PARTIAL_FAKE     – Suspicious windows exist; manipulation covers a segment.
    INCONCLUSIVE     – Scores are too scattered to confidently categorise.

    Thresholds were chosen empirically — adjust them if your validation
    data shows systematic mis-classification of a particular pattern type.
    """
    mean_score = float(np.mean(smoothed))
    variance   = float(np.var(smoothed))

    if mean_score >= 0.60 and variance < 0.05:
        return "CONSISTENT_FAKE"

    if mean_score < 0.40 and variance < 0.05:
        return "CONSISTENT_REAL"

    if suspicious_windows and suspicious_frame_ratio >= 0.20:
        return "PARTIAL_FAKE"

    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_temporal(
    frame_indices: List[int],
    prob_fakes: List[float],
    smooth_window: int = 5,
    suspicion_threshold: float = 0.5,
    min_window_length: int = 2,
) -> TemporalAnalysis:
    """
    Full temporal analysis of per-frame deepfake scores.

    Args:
        frame_indices:        Frame numbers corresponding to each score.
        prob_fakes:           P(fake) per frame, from inference_result.json.
        smooth_window:        Gaussian window size in frames.
        suspicion_threshold:  Score above which a frame is suspicious.
        min_window_length:    Minimum consecutive suspicious frames to form a window.

    Returns:
        TemporalAnalysis with all statistics and the temporal verdict label.
    """
    if not frame_indices:
        return TemporalAnalysis(
            frame_indices=[], raw_scores=[], smoothed_scores=[],
            suspicious_windows=[], score_variance=0.0,
            peak_frame_idx=-1, peak_score=0.0,
            suspicious_frame_ratio=0.0, run_length_score=1.0,
            temporal_verdict="INCONCLUSIVE",
        )

    # Sort by frame index — inference output order is not guaranteed
    paired = sorted(zip(frame_indices, prob_fakes), key=lambda t: t[0])
    sorted_indices = [t[0] for t in paired]
    raw_scores     = np.array([t[1] for t in paired], dtype=np.float32)

    smoothed = _gaussian_smooth(raw_scores, window_size=smooth_window)

    peak_pos               = int(np.argmax(raw_scores))
    suspicious_frame_ratio = float(np.mean(raw_scores >= suspicion_threshold))
    suspicious_windows     = _find_suspicious_windows(
        sorted_indices, smoothed, suspicion_threshold, min_window_length
    )
    run_length_score       = _compute_run_length_score(raw_scores, suspicion_threshold)
    temporal_verdict       = _classify_temporal_pattern(
        smoothed, suspicious_windows, suspicious_frame_ratio
    )

    return TemporalAnalysis(
        frame_indices          = sorted_indices,
        raw_scores             = [round(float(s), 4) for s in raw_scores],
        smoothed_scores        = [round(float(s), 4) for s in smoothed],
        suspicious_windows     = suspicious_windows,
        score_variance         = round(float(np.var(raw_scores)), 4),
        peak_frame_idx         = sorted_indices[peak_pos],
        peak_score             = round(float(raw_scores[peak_pos]), 4),
        suspicious_frame_ratio = round(suspicious_frame_ratio, 4),
        run_length_score       = run_length_score,
        temporal_verdict       = temporal_verdict,
    )