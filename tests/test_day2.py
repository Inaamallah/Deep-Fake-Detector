# tests/test_day2.py
import numpy as np
import pytest
import cv2

from src.detection.face_aligner import align_face, expand_bbox, REFERENCE_LANDMARKS_112
from src.detection.face_detector import DetectedFace
from src.detection.quality_filter import assess_quality


def _make_face(bbox=(100, 100, 200, 200), confidence=0.95) -> DetectedFace:
    """Helper: realistic 5-landmark face at the given bbox."""
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    landmarks = [
        (cx - 30, cy - 20),   # left eye
        (cx + 30, cy - 20),   # right eye
        (cx,      cy + 5),    # nose
        (cx - 20, cy + 30),   # mouth left
        (cx + 20, cy + 30),   # mouth right
    ]
    return DetectedFace(bbox=bbox, landmarks=landmarks,
                        confidence=confidence, frame_idx=0, face_idx=0)


def _make_frame(h=640, w=640) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw a white square where the "face" is
    frame[100:300, 100:300] = 200
    return frame


class TestAlignment:
    def test_output_shape(self):
        frame = _make_frame()
        face = _make_face()
        crop = align_face(frame, face, output_size=112)
        assert crop is not None
        assert crop.shape == (112, 112, 3)

    def test_custom_output_size(self):
        frame = _make_frame()
        face = _make_face()
        crop = align_face(frame, face, output_size=224)
        assert crop.shape == (224, 224, 3)

    def test_insufficient_landmarks_returns_none(self):
        frame = _make_frame()
        face = _make_face()
        face.landmarks = face.landmarks[:3]   # only 3 points — too few
        crop = align_face(frame, face, output_size=112)
        assert crop is None

    def test_expand_bbox_clamps(self):
        x, y, w, h = expand_bbox((10, 10, 100, 100), frame_h=200, frame_w=200, padding=1.0)
        assert x >= 0 and y >= 0
        assert x + w <= 200 and y + h <= 200


class TestQualityFilter:
    def test_sharp_large_face_passes(self):
        # High-frequency (checkerboard) crop → high Laplacian variance
        crop = np.tile(
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
            (56, 56, 1)
        ).repeat(3, axis=2)[:112, :112, :]
        face = _make_face(bbox=(0, 0, 150, 150))
        report = assess_quality(crop, face)
        assert report.passed

    def test_tiny_face_fails(self):
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        face = _make_face(bbox=(0, 0, 30, 30))  # 30px — below 60px threshold
        report = assess_quality(crop, face)
        assert not report.passed
        assert "too_small" in report.rejection_reason

    def test_low_confidence_fails(self):
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        face = _make_face(confidence=0.3)
        report = assess_quality(crop, face)
        assert not report.passed
        assert "low_confidence" in report.rejection_reason

    def test_blurry_face_fails(self):
        # Solid-color crop → near-zero Laplacian variance
        crop = np.full((112, 112, 3), 128, dtype=np.uint8)
        face = _make_face()
        report = assess_quality(crop, face)
        assert not report.passed
        assert "too_blurry" in report.rejection_reason