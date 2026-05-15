# src/scoring/confidence_estimator.py  ← NEW FILE
"""
Bootstrap confidence intervals for the video-level P(fake) estimate.

Why bootstrap rather than a parametric interval (e.g. a normal CI)?
  - We make no assumption about the distribution of frame scores.
  - The distribution is often bimodal (many frames near 0 or near 1)
    or skewed, violating the normality assumption.
  - Bootstrap is computationally cheap for the number of samples we have
    (typically 20–120 face crops per video).
  - The resulting interval has a clear, honest interpretation: if we had
    scored a different random sample of frames from this video, the mean
    P(fake) would fall in [lower, upper] 95% of the time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceEstimate:
    """Bootstrap confidence interval and supporting statistics."""
    point_estimate: float   # simple mean of all per-face P(fake) scores
    ci_lower_95:    float   # 2.5th  percentile of bootstrap distribution
    ci_upper_95:    float   # 97.5th percentile
    ci_lower_90:    float   # 5th    percentile
    ci_upper_90:    float   # 95th   percentile
    bootstrap_std:  float   # std of bootstrap sample means (standard error)
    n_samples_used: int     # number of face scores used
    interpretation: str     # human-readable summary


def _bootstrap_mean(
    scores: np.ndarray,
    n_iterations: int,
    seed: int,
) -> np.ndarray:
    """
    Generate a bootstrap distribution of the mean score.

    For each of n_iterations iterations:
      - Sample len(scores) values from scores WITH replacement.
      - Record their mean.

    The resulting array of n_iterations means approximates the sampling
    distribution of the estimator under repeated sampling.

    Why 2000 iterations?
    The 2.5th and 97.5th percentiles of a 2000-element array are each
    estimated from ~50 extreme values — enough for stability without
    being computationally wasteful. 500 iterations gives noticeably
    noisier intervals; 5000 gives no practical improvement.
    """
    rng        = np.random.default_rng(seed)
    n          = len(scores)
    boot_means = np.empty(n_iterations, dtype=np.float32)

    for i in range(n_iterations):
        sample       = rng.choice(scores, size=n, replace=True)
        boot_means[i] = float(sample.mean())

    return boot_means


def _interpret(
    ci_lower: float,
    ci_upper: float,
    point_estimate: float,
    verdict: str,
) -> str:
    width = ci_upper - ci_lower

    if width < 0.10:
        certainty = "very high certainty"
    elif width < 0.20:
        certainty = "moderate certainty"
    elif width < 0.30:
        certainty = "low certainty"
    else:
        certainty = "very uncertain"

    return (
        f"The model predicts '{verdict}' with {certainty}. "
        f"Point estimate P(fake) = {point_estimate:.3f}. "
        f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}] "
        f"(interval width {width:.3f}). "
        f"{'Wide interval — collect more video footage for a reliable result.' if width >= 0.30 else ''}"
    ).strip()


def estimate_confidence(
    prob_fakes: list,
    verdict: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> ConfidenceEstimate:
    """
    Compute bootstrap confidence intervals for the video-level P(fake).

    Args:
        prob_fakes:  Per-face P(fake) scores from inference_result.json.
        verdict:     "REAL" or "DEEPFAKE" — used in the interpretation string.
        n_bootstrap: Number of bootstrap iterations (2000 is a safe default).
        seed:        RNG seed for reproducibility.

    Returns:
        ConfidenceEstimate with 90% and 95% intervals.
    """
    if not prob_fakes:
        return ConfidenceEstimate(
            point_estimate=0.5,
            ci_lower_95=0.0, ci_upper_95=1.0,
            ci_lower_90=0.0, ci_upper_90=1.0,
            bootstrap_std=0.5,
            n_samples_used=0,
            interpretation="No face scores — cannot estimate confidence.",
        )

    scores         = np.array(prob_fakes, dtype=np.float32)
    point_estimate = float(scores.mean())

    boot_means = _bootstrap_mean(scores, n_iterations=n_bootstrap, seed=seed)

    ci_lower_95 = float(np.percentile(boot_means, 2.5))
    ci_upper_95 = float(np.percentile(boot_means, 97.5))
    ci_lower_90 = float(np.percentile(boot_means, 5.0))
    ci_upper_90 = float(np.percentile(boot_means, 95.0))
    bootstrap_std = float(np.std(boot_means))

    return ConfidenceEstimate(
        point_estimate = round(point_estimate, 4),
        ci_lower_95    = round(ci_lower_95, 4),
        ci_upper_95    = round(ci_upper_95, 4),
        ci_lower_90    = round(ci_lower_90, 4),
        ci_upper_90    = round(ci_upper_90, 4),
        bootstrap_std  = round(bootstrap_std, 4),
        n_samples_used = len(scores),
        interpretation = _interpret(ci_lower_95, ci_upper_95, point_estimate, verdict),
    )