# tests/test_day4.py  ← NEW FILE
"""
Day 4 unit tests.
Run: pytest tests/test_day4.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import cv2

from src.scoring.temporal_analyzer import (
    _gaussian_smooth,
    _find_suspicious_windows,
    _compute_run_length_score,
    _classify_temporal_pattern,
    analyze_temporal,
    SuspiciousWindow,
    TemporalAnalysis,
)
from src.scoring.confidence_estimator import (
    _bootstrap_mean,
    estimate_confidence,
    ConfidenceEstimate,
)


# ===========================================================================
# Temporal analyser tests
# ===========================================================================

class TestGaussianSmooth:
    def test_flat_signal_unchanged(self):
        scores = np.full(20, 0.8, dtype=np.float32)
        result = _gaussian_smooth(scores, window_size=5)
        np.testing.assert_allclose(result, 0.8, atol=1e-3)

    def test_output_length_preserved(self):
        for n in [3, 10, 50, 120]:
            scores = np.random.rand(n).astype(np.float32)
            result = _gaussian_smooth(scores, window_size=5)
            assert len(result) == n, f"Length mismatch for n={n}"

    def test_spike_is_smoothed(self):
        scores = np.zeros(11, dtype=np.float32)
        scores[5] = 1.0   # single spike at centre
        result = _gaussian_smooth(scores, window_size=5)
        # The spike should spread to neighbours, reducing the peak
        assert result[5] < 1.0
        # And raise the values on either side above zero
        assert result[4] > 0.0
        assert result[6] > 0.0

    def test_single_element(self):
        scores = np.array([0.7], dtype=np.float32)
        result = _gaussian_smooth(scores, window_size=5)
        assert result.shape == (1,)

    def test_values_stay_bounded(self):
        scores = np.random.rand(100).astype(np.float32)
        result = _gaussian_smooth(scores, window_size=5)
        # Smoothing a [0,1] signal should stay in [0,1]
        assert float(result.min()) >= -1e-4
        assert float(result.max()) <= 1.0 + 1e-4


class TestFindSuspiciousWindows:
    def test_single_contiguous_block(self):
        indices = list(range(10))
        scores  = np.array(
            [0.1, 0.1, 0.8, 0.9, 0.85, 0.7, 0.1, 0.1, 0.1, 0.1],
            dtype=np.float32
        )
        windows = _find_suspicious_windows(indices, scores, threshold=0.5, min_length=2)
        assert len(windows) == 1
        assert windows[0].start_frame == 2
        assert windows[0].end_frame   == 5
        assert windows[0].frame_count == 4

    def test_single_frame_spike_filtered_out(self):
        indices = list(range(5))
        scores  = np.array([0.1, 0.9, 0.1, 0.1, 0.1], dtype=np.float32)
        # min_length=2 means isolated single frames are ignored
        windows = _find_suspicious_windows(indices, scores, threshold=0.5, min_length=2)
        assert len(windows) == 0

    def test_window_touching_end_of_video(self):
        indices = list(range(5))
        scores  = np.array([0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float32)
        windows = _find_suspicious_windows(indices, scores, threshold=0.5, min_length=2)
        assert len(windows) == 1
        assert windows[0].end_frame == 4

    def test_all_real_no_windows(self):
        indices = list(range(20))
        scores  = np.zeros(20, dtype=np.float32)
        windows = _find_suspicious_windows(indices, scores, threshold=0.5, min_length=2)
        assert len(windows) == 0

    def test_two_separate_windows(self):
        indices = list(range(12))
        scores  = np.array(
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
            dtype=np.float32
        )
        windows = _find_suspicious_windows(indices, scores, threshold=0.5, min_length=2)
        assert len(windows) == 2

    def test_window_stats_are_correct(self):
        indices = [0, 1, 2]
        scores  = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        windows = _find_suspicious_windows(indices, scores, threshold=0.5, min_length=1)
        assert len(windows) == 1
        assert abs(windows[0].mean_prob_fake - 0.8) < 1e-3
        assert abs(windows[0].max_prob_fake  - 0.9) < 1e-3


class TestRunLengthScore:
    def test_all_real_returns_one(self):
        scores = np.zeros(20, dtype=np.float32)
        assert _compute_run_length_score(scores, threshold=0.5) == 1.0

    def test_all_fake_returns_one(self):
        scores = np.ones(20, dtype=np.float32)
        assert _compute_run_length_score(scores, threshold=0.5) == 1.0

    def test_clustered_fakes_score_above_one(self):
        # 10 consecutive fakes in the middle of 20 frames → highly clustered
        scores = np.array(
            [0.0]*5 + [1.0]*10 + [0.0]*5, dtype=np.float32
        )
        score = _compute_run_length_score(scores, threshold=0.5)
        assert score > 1.0, f"Expected > 1.0, got {score}"

    def test_scattered_fakes_score_near_one(self):
        # Alternating 0, 1, 0, 1, ... → minimal run length
        scores = np.array([0.0, 1.0] * 10, dtype=np.float32)
        score = _compute_run_length_score(scores, threshold=0.5)
        # With p_fake=0.5 and mean_run=1, ratio ≈ 1/2 = 0.5
        assert score < 1.0, f"Expected < 1.0, got {score}"


class TestClassifyTemporalPattern:
    def _make_windows(self, n: int) -> list:
        return [MagicMock() for _ in range(n)]

    def test_consistent_fake(self):
        smoothed = np.full(20, 0.85, dtype=np.float32)
        verdict  = _classify_temporal_pattern(smoothed, [], 0.9)
        assert verdict == "CONSISTENT_FAKE"

    def test_consistent_real(self):
        smoothed = np.full(20, 0.1, dtype=np.float32)
        verdict  = _classify_temporal_pattern(smoothed, [], 0.05)
        assert verdict == "CONSISTENT_REAL"

    def test_partial_fake(self):
        # High variance, some suspicious windows, ratio above 0.2
        smoothed = np.array([0.1]*8 + [0.9]*4 + [0.1]*8, dtype=np.float32)
        windows  = self._make_windows(1)
        verdict  = _classify_temporal_pattern(smoothed, windows, 0.25)
        assert verdict == "PARTIAL_FAKE"

    def test_inconclusive_when_no_windows_low_ratio(self):
        smoothed = np.array([0.3, 0.6, 0.2, 0.7, 0.4], dtype=np.float32)
        verdict  = _classify_temporal_pattern(smoothed, [], 0.10)
        assert verdict == "INCONCLUSIVE"


class TestAnalyzeTemporal:
    def test_empty_input(self):
        result = analyze_temporal([], [])
        assert result.temporal_verdict == "INCONCLUSIVE"
        assert result.frame_indices == []

    def test_sorts_by_frame_index(self):
        # Pass frames out of order — should come back sorted
        indices = [10, 2, 7, 1]
        probs   = [0.9, 0.1, 0.5, 0.2]
        result  = analyze_temporal(indices, probs)
        assert result.frame_indices == sorted(indices)

    def test_output_lengths_match(self):
        n       = 30
        indices = list(range(n))
        probs   = np.random.rand(n).tolist()
        result  = analyze_temporal(indices, probs)
        assert len(result.raw_scores)     == n
        assert len(result.smoothed_scores) == n
        assert len(result.frame_indices)  == n

    def test_peak_frame_is_highest_scorer(self):
        indices = [0, 1, 2, 3, 4]
        probs   = [0.1, 0.2, 0.95, 0.3, 0.1]
        result  = analyze_temporal(indices, probs)
        assert result.peak_frame_idx == 2
        assert abs(result.peak_score - 0.95) < 1e-4

    def test_to_dict_is_serialisable(self):
        indices = list(range(10))
        probs   = np.random.rand(10).tolist()
        result  = analyze_temporal(indices, probs)
        # Should not raise
        serialised = json.dumps(result.to_dict())
        parsed     = json.loads(serialised)
        assert parsed["temporal_verdict"] in (
            "CONSISTENT_FAKE", "CONSISTENT_REAL", "PARTIAL_FAKE", "INCONCLUSIVE"
        )


# ===========================================================================
# Confidence estimator tests
# ===========================================================================

class TestBootstrapMean:
    def test_output_length(self):
        scores     = np.random.rand(50).astype(np.float32)
        boot_means = _bootstrap_mean(scores, n_iterations=500, seed=42)
        assert len(boot_means) == 500

    def test_constant_scores_give_zero_variance(self):
        scores     = np.full(50, 0.7, dtype=np.float32)
        boot_means = _bootstrap_mean(scores, n_iterations=500, seed=42)
        # Every bootstrap sample is all 0.7 → every mean is 0.7
        np.testing.assert_allclose(boot_means, 0.7, atol=1e-5)

    def test_reproducible_with_same_seed(self):
        scores = np.random.rand(50).astype(np.float32)
        b1     = _bootstrap_mean(scores, n_iterations=200, seed=99)
        b2     = _bootstrap_mean(scores, n_iterations=200, seed=99)
        np.testing.assert_array_equal(b1, b2)

    def test_different_seeds_differ(self):
        scores = np.random.rand(50).astype(np.float32)
        b1     = _bootstrap_mean(scores, n_iterations=200, seed=1)
        b2     = _bootstrap_mean(scores, n_iterations=200, seed=2)
        assert not np.array_equal(b1, b2)


class TestEstimateConfidence:
    def test_empty_input(self):
        result = estimate_confidence([], verdict="REAL")
        assert result.n_samples_used == 0
        assert result.ci_lower_95 == 0.0
        assert result.ci_upper_95 == 1.0

    def test_all_fake_scores(self):
        scores = [0.95] * 50
        result = estimate_confidence(scores, verdict="DEEPFAKE")
        # With all scores near 0.95, the CI should be narrow and high
        assert result.point_estimate > 0.9
        assert result.ci_lower_95    > 0.8
        assert result.ci_upper_95    <= 1.0 + 1e-4

    def test_all_real_scores(self):
        scores = [0.05] * 50
        result = estimate_confidence(scores, verdict="REAL")
        assert result.point_estimate < 0.1
        assert result.ci_upper_95    < 0.2

    def test_ci_ordering(self):
        scores = np.random.rand(80).tolist()
        result = estimate_confidence(scores, verdict="REAL")
        assert result.ci_lower_95 < result.ci_upper_95
        assert result.ci_lower_90 < result.ci_upper_90
        # 90% CI is narrower than 95% CI
        width_95 = result.ci_upper_95 - result.ci_lower_95
        width_90 = result.ci_upper_90 - result.ci_lower_90
        assert width_90 <= width_95

    def test_n_samples_recorded(self):
        scores = [0.5] * 37
        result = estimate_confidence(scores, verdict="REAL")
        assert result.n_samples_used == 37

    def test_interpretation_is_string(self):
        scores = np.random.rand(30).tolist()
        result = estimate_confidence(scores, verdict="DEEPFAKE")
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 10

    def test_wide_interval_mentions_more_footage(self):
        # Bimodal, high-variance scores → wide interval
        scores = [0.05] * 10 + [0.95] * 10
        result = estimate_confidence(scores, verdict="DEEPFAKE", n_bootstrap=2000)
        if (result.ci_upper_95 - result.ci_lower_95) >= 0.30:
            assert "footage" in result.interpretation.lower()