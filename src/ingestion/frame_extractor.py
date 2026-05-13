# src/ingestion/frame_extractor.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np
from tqdm import tqdm

from config import settings
from src.utils.logger import logger


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""


@dataclass
class FrameBatch:
    """
    Output of frame extraction for one video.
    frames_dir contains numbered PNG files: 000000.png, 000001.png, ...
    """
    video_id: str
    frames_dir: Path
    frame_paths: List[Path]
    total_frames_in_video: int
    video_duration_seconds: float
    extraction_strategy: str   # "scene_change" | "uniform" | "hybrid"
    elapsed_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.frame_paths)

    def __str__(self) -> str:
        return (
            f"FrameBatch(video={self.video_id[:8]}… "
            f"frames={self.count} "
            f"strategy={self.extraction_strategy} "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )


def _frame_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Mean absolute pixel difference between two grayscale frames.
    Returns a value 0-255. Higher = more visual change.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(gray_a - gray_b)))


def _resize_frame(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Letterbox-resize frame to target_size while preserving aspect ratio.
    Pads with black bars.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    # Scale to fit inside target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas and paste
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def _scene_change_indices(
    cap: cv2.VideoCapture,
    total_frames: int,
    threshold: float,
    max_frames: int,
    log,
) -> List[int]:
    """
    First pass: read every frame, record indices where scene change > threshold.
    If too many scene-change frames are found, subsample them evenly.
    If too few, fill in uniform samples.
    """
    selected_indices: List[int] = [0]  # always include first frame
    prev_frame = None
    diff_scores: List[float] = []

    # Read every Nth frame for efficiency on long videos
    # We never want to read more than ~1000 frames in the first pass
    stride = max(1, total_frames // 1000)

    log.info("scene_detection_pass", stride=stride, total_frames=total_frames)

    for idx in range(0, total_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if prev_frame is not None:
            diff = _frame_diff(prev_frame, frame)
            diff_scores.append(diff)
            if diff >= threshold:
                selected_indices.append(idx)

        prev_frame = frame

    # Always include the last frame
    selected_indices.append(total_frames - 1)
    selected_indices = sorted(set(selected_indices))

    log.info(
        "scene_detection_complete",
        raw_scenes=len(selected_indices),
        avg_diff=round(float(np.mean(diff_scores)) if diff_scores else 0, 2),
    )

    # If we got way too many scene changes, subsample evenly
    if len(selected_indices) > max_frames:
        step = len(selected_indices) / max_frames
        selected_indices = [
            selected_indices[int(i * step)] for i in range(max_frames)
        ]

    # If we got too few, add uniform samples to reach min_frames
    if len(selected_indices) < settings.min_frames:
        uniform = [
            int(i * total_frames / settings.min_frames)
            for i in range(settings.min_frames)
        ]
        selected_indices = sorted(set(selected_indices + uniform))

    return selected_indices[:max_frames]


def _uniform_indices(
    total_frames: int,
    fps: float,
    target_fps: float,
    max_frames: int,
) -> List[int]:
    """
    Evenly sample frames at target_fps.
    E.g. a 300-frame / 30fps video with target_fps=2.0 → every 15th frame.
    """
    step = max(1, int(fps / target_fps))
    indices = list(range(0, total_frames, step))
    if len(indices) > max_frames:
        subsample_step = len(indices) / max_frames
        indices = [indices[int(i * subsample_step)] for i in range(max_frames)]
    return indices


def extract_frames(
    video_path: Path,
    video_id: str,
    duration_seconds: float,
    fps: float,
    strategy: str = "hybrid",
) -> FrameBatch:
    """
    Extract frames from a video file.

    Args:
        video_path:        Path to the video file.
        video_id:          Stable ID string (used to name output directory).
        duration_seconds:  Video duration (from VideoMetadata).
        fps:               Video FPS (from VideoMetadata).
        strategy:          "scene_change" | "uniform" | "hybrid"
                           hybrid = scene_change for short clips,
                                    uniform for long ones (>60s talking head).

    Returns:
        FrameBatch with paths to all saved PNG files.
    """
    log = logger.bind(video_id=video_id, strategy=strategy)
    start = time.time()

    output_dir = settings.raw_frames_dir / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FrameExtractionError(f"OpenCV could not open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Some containers don't report frame count — estimate from duration
        total_frames = max(1, int(duration_seconds * fps))
        log.warning("frame_count_unavailable_estimating", estimated=total_frames)

    log.info(
        "extraction_starting",
        total_frames=total_frames,
        duration=duration_seconds,
        fps=fps,
        output_dir=str(output_dir),
    )

    # ── Choose strategy ──────────────────────────────────────────────────────
    if strategy == "hybrid":
        # Talking-head / interview videos are often static and benefit from
        # uniform sampling; action videos benefit from scene-change detection.
        actual_strategy = "uniform" if duration_seconds > 60 else "scene_change"
    else:
        actual_strategy = strategy

    if actual_strategy == "scene_change":
        indices = _scene_change_indices(
            cap, total_frames, settings.scene_threshold,
            settings.max_frames, log
        )
    else:
        indices = _uniform_indices(
            total_frames, fps, settings.target_fps, settings.max_frames
        )

    log.info("frame_indices_selected", count=len(indices), strategy=actual_strategy)

    # ── Extract and save frames ──────────────────────────────────────────────
    frame_paths: List[Path] = []
    failed = 0

    for save_idx, frame_idx in enumerate(
        tqdm(indices, desc="Extracting frames", unit="frame", leave=False)
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning("frame_read_failed", frame_idx=frame_idx)
            failed += 1
            continue

        # Resize to standard size
        frame = _resize_frame(frame, settings.frame_size)

        # Save as PNG (lossless — important for artifact detection later)
        save_path = output_dir / f"{save_idx:06d}.png"
        success = cv2.imwrite(str(save_path), frame)
        if success:
            frame_paths.append(save_path)
        else:
            log.error("frame_save_failed", path=str(save_path))
            failed += 1

    cap.release()

    elapsed = time.time() - start

    if not frame_paths:
        raise FrameExtractionError(
            f"No frames were successfully extracted from {video_path}. "
            f"Attempted {len(indices)} frames, {failed} failed."
        )

    log.info(
        "extraction_complete",
        extracted=len(frame_paths),
        failed=failed,
        elapsed_s=round(elapsed, 2),
        output_dir=str(output_dir),
    )

    return FrameBatch(
        video_id=video_id,
        frames_dir=output_dir,
        frame_paths=frame_paths,
        total_frames_in_video=total_frames,
        video_duration_seconds=duration_seconds,
        extraction_strategy=actual_strategy,
        elapsed_seconds=elapsed,
    )