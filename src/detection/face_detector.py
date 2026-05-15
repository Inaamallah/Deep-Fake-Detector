# src/detection/face_detector.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.utils.logger import logger


@dataclass
class DetectedFace:
    """
    One detected face in one frame.
    bbox:       (x, y, w, h) in pixel coords — top-left origin
    landmarks:  5 key points [(x,y), ...] — left_eye, right_eye,
                nose_tip, mouth_left, mouth_right
    confidence: detector confidence score 0–1
    frame_idx:  which frame this came from
    face_idx:   which face in the frame (0-indexed, for multi-face frames)
    """
    bbox: Tuple[int, int, int, int]
    landmarks: List[Tuple[float, float]]
    confidence: float
    frame_idx: int
    face_idx: int

    @property
    def area(self) -> int:
        _, _, w, h = self.bbox
        return w * h


class MediaPipeDetector:
    """
    Primary detector. Fast, runs well on CPU.
    Uses FaceDetection (not FaceMesh) — we only need bbox + 6 keypoints,
    not the full 468-point mesh. Detection is ~5-10ms per frame on CPU.
    """

    # MediaPipe FaceDetection keypoint indices for our 5 points
    # (left_eye, right_eye, nose_tip, mouth_left, mouth_right)
    _KP_IDX = [0, 1, 2, 3, 4]

    def __init__(self, min_confidence: float = 0.6):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Installed mediapipe package does not expose the legacy "
                "`solutions` API required by MediaPipeDetector."
            )

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,          # model 1 = full-range (up to 5m)
            min_detection_confidence=min_confidence,
        )
        self.log = logger.bind(detector="mediapipe")

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[DetectedFace]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return []

        faces = []
        for face_idx, det in enumerate(results.detections):
            score = det.score[0] if det.score else 0.0
            bb = det.location_data.relative_bounding_box

            # Convert relative coords → absolute pixels
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width * w), w - x)
            bh = min(int(bb.height * h), h - y)

            # Extract 5 landmark keypoints
            kps = det.location_data.relative_keypoints
            landmarks = [
                (kps[i].x * w, kps[i].y * h)
                for i in self._KP_IDX
            ]

            faces.append(DetectedFace(
                bbox=(x, y, bw, bh),
                landmarks=landmarks,
                confidence=float(score),
                frame_idx=frame_idx,
                face_idx=face_idx,
            ))

        return faces

    def close(self):
        self._detector.close()


class MTCNNDetector:
    """
    Fallback detector. Slower (~80ms/frame on CPU) but handles
    harder cases: profile angles, low light, partial occlusion.
    Only called when MediaPipe returns nothing.
    """

    def __init__(self, min_confidence: float = 0.85):
        # Lazy import — only load MTCNN when actually needed
        from facenet_pytorch import MTCNN
        import torch
        self._mtcnn = MTCNN(
            keep_all=True,
            device="cpu",
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.85],
            post_process=False,
        )
        self.min_confidence = min_confidence
        self.log = logger.bind(detector="mtcnn")

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[DetectedFace]:
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)
        except Exception as e:
            self.log.warning("mtcnn_detect_failed", error=str(e))
            return []

        if boxes is None or probs is None:
            return []

        faces = []
        for face_idx, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
            if prob < self.min_confidence:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            if w <= 0 or h <= 0:
                continue

            # MTCNN gives 5 landmarks in same order as our convention
            # (left_eye, right_eye, nose, mouth_left, mouth_right)
            landmarks_list = [(float(p[0]), float(p[1])) for p in lm]

            faces.append(DetectedFace(
                bbox=(x1, y1, w, h),
                landmarks=landmarks_list,
                confidence=float(prob),
                frame_idx=frame_idx,
                face_idx=face_idx,
            ))

        return faces


class FaceDetector:
    """
    Unified detector: tries MediaPipe first, falls back to MTCNN.
    This is the only class the rest of the pipeline imports.
    """

    def __init__(
        self,
        mp_confidence: float = 0.6,
        mtcnn_confidence: float = 0.85,
    ):
        self.log = logger.bind(component="face_detector")
        self._mp: Optional[MediaPipeDetector]
        try:
            self._mp = MediaPipeDetector(min_confidence=mp_confidence)
        except Exception as e:
            self._mp = None
            self.log.warning(
                "mediapipe_unavailable_using_mtcnn",
                error=str(e),
            )

        self._mtcnn: Optional[MTCNNDetector] = None  # lazy init
        self._mtcnn_confidence = mtcnn_confidence

        # Stats for logging
        self._mp_hits = 0
        self._mtcnn_hits = 0
        self._misses = 0

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[DetectedFace]:
        faces: List[DetectedFace] = []
        if self._mp is not None:
            faces = self._mp.detect(frame, frame_idx)

            if faces:
                self._mp_hits += 1
                return faces

        # MediaPipe found nothing — try MTCNN
        if self._mtcnn is None:
            self.log.info("initialising_mtcnn_fallback")
            self._mtcnn = MTCNNDetector(min_confidence=self._mtcnn_confidence)

        faces = self._mtcnn.detect(frame, frame_idx)

        if faces:
            self._mtcnn_hits += 1
        else:
            self._misses += 1

        return faces

    def stats(self) -> dict:
        total = self._mp_hits + self._mtcnn_hits + self._misses
        return {
            "total_frames": total,
            "mediapipe_hits": self._mp_hits,
            "mtcnn_hits": self._mtcnn_hits,
            "misses": self._misses,
            "mediapipe_available": self._mp is not None,
            "detection_rate": round(
                (self._mp_hits + self._mtcnn_hits) / total, 3
            ) if total else 0.0,
        }

    def close(self):
        if self._mp is not None:
            self._mp.close()
