# src/detection/quality_filter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.detection.face_detector import DetectedFace


@dataclass
class QualityReport:
    passed: bool
    blur_score: float           # Laplacian variance — higher = sharper
    face_size_px: int           # min(width, height) of bbox
    confidence: float
    rejection_reason: Optional[str] = None


# Tunable thresholds
MIN_FACE_SIZE_PX = 60          # faces smaller than 60px are too small for the model
MIN_BLUR_SCORE = 50.0          # Laplacian variance below this = too blurry
MIN_CONFIDENCE = 0.50          # detector confidence floor


def _laplacian_blur_score(crop: np.ndarray) -> float:
    """
    Laplacian variance — the standard no-reference sharpness metric.
    Blurry images have low frequency content → low Laplacian variance.
    Sharp images have strong edges → high variance.
    Rule of thumb: < 50 = too blurry, > 100 = clearly sharp.
    """
    if crop.ndim == 2:
        gray = crop
    elif crop.ndim == 3 and crop.shape[2] == 1:
        gray = crop[:, :, 0]
    elif crop.ndim == 3 and crop.shape[2] in (3, 4):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    elif crop.ndim == 3:
        gray = crop[:, :, 0]
    else:
        raise ValueError(f"Expected a 2D or 3D image crop, got shape {crop.shape}")

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_quality(
    crop: np.ndarray,
    face: DetectedFace,
    min_face_size: int = MIN_FACE_SIZE_PX,
    min_blur: float = MIN_BLUR_SCORE,
    min_confidence: float = MIN_CONFIDENCE,
) -> QualityReport:
    """
    Assess whether a face crop is worth passing to the model.

    Checks (in order, fail-fast):
    1. Detector confidence
    2. Face bounding box size
    3. Sharpness (Laplacian variance)
    """
    _, _, bw, bh = face.bbox
    face_size = min(bw, bh)
    blur_score = _laplacian_blur_score(crop)

    if face.confidence < min_confidence:
        return QualityReport(
            passed=False,
            blur_score=blur_score,
            face_size_px=face_size,
            confidence=face.confidence,
            rejection_reason=f"low_confidence:{face.confidence:.2f}",
        )

    if face_size < min_face_size:
        return QualityReport(
            passed=False,
            blur_score=blur_score,
            face_size_px=face_size,
            confidence=face.confidence,
            rejection_reason=f"too_small:{face_size}px",
        )

    if blur_score < min_blur:
        return QualityReport(
            passed=False,
            blur_score=blur_score,
            face_size_px=face_size,
            confidence=face.confidence,
            rejection_reason=f"too_blurry:{blur_score:.1f}",
        )

    return QualityReport(
        passed=True,
        blur_score=blur_score,
        face_size_px=face_size,
        confidence=face.confidence,
    )
