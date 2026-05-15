# src/detection/face_extractor.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm

from config import settings
from src.detection.face_aligner import align_face
from src.detection.face_detector import DetectedFace, FaceDetector
from src.detection.quality_filter import QualityReport, assess_quality
from src.utils.logger import logger


@dataclass
class FaceCropRecord:
    """Metadata for one saved face crop."""
    frame_idx: int
    face_idx: int
    frame_path: str
    crop_path: str
    bbox: tuple
    confidence: float
    blur_score: float
    face_size_px: int


@dataclass
class FaceBatch:
    """Output of the full face extraction run for one video."""
    video_id: str
    crops_dir: Path
    records: List[FaceCropRecord] = field(default_factory=list)
    rejected: int = 0
    frames_with_no_face: int = 0
    elapsed_seconds: float = 0.0
    detector_stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crops_dir"] = str(self.crops_dir)
        d["records"] = [
            {**r, "crop_path": str(r["crop_path"]),
             "frame_path": str(r["frame_path"])}
            for r in d["records"]
        ]
        return d

    def __str__(self) -> str:
        return (
            f"FaceBatch(video={self.video_id[:8]}… "
            f"crops={self.count} rejected={self.rejected} "
            f"no_face_frames={self.frames_with_no_face} "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )


def extract_faces_from_manifest(
    manifest_path: Path,
    output_size: int = 112,
    max_faces_per_frame: int = 3,
) -> FaceBatch:
    """
    Full Day 2 pipeline for one video.

    Args:
        manifest_path:       Path to the manifest.json written by Day 1.
        output_size:         Aligned crop side length in pixels (default 112).
        max_faces_per_frame: Cap faces per frame (avoids wasting budget on crowds).

    Returns:
        FaceBatch with one record per saved crop + face_manifest.json on disk.
    """
    manifest = json.loads(manifest_path.read_text())
    video_id = manifest["video_id"]
    frame_paths = [Path(p) for p in manifest["frame_paths"]]

    log = logger.bind(video_id=video_id, total_frames=len(frame_paths))
    log.info("face_extraction_starting")

    crops_dir = settings.face_crops_dir / video_id
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    batch = FaceBatch(video_id=video_id, crops_dir=crops_dir)
    start = time.time()

    for frame_path in tqdm(frame_paths, desc="Detecting faces", unit="frame", leave=False):
        # Parse frame index from filename (e.g. 000042.png → 42)
        try:
            frame_idx = int(frame_path.stem)
        except ValueError:
            frame_idx = frame_paths.index(frame_path)

        frame = cv2.imread(str(frame_path))
        if frame is None:
            log.warning("frame_unreadable", path=str(frame_path))
            continue

        faces = detector.detect(frame, frame_idx)

        if not faces:
            batch.frames_with_no_face += 1
            continue

        # Limit faces per frame to avoid crowd scenes dominating the budget
        faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        faces = faces[:max_faces_per_frame]

        for face in faces:
            # 1. Align
            crop = align_face(frame, face, output_size=output_size)
            if crop is None:
                batch.rejected += 1
                continue

            # 2. Quality filter
            report = assess_quality(crop, face)
            if not report.passed:
                log.debug(
                    "face_rejected",
                    frame_idx=frame_idx,
                    reason=report.rejection_reason,
                )
                batch.rejected += 1
                continue

            # 3. Save
            crop_filename = f"frame_{frame_idx:06d}_face_{face.face_idx}.png"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            batch.records.append(FaceCropRecord(
                frame_idx=frame_idx,
                face_idx=face.face_idx,
                frame_path=str(frame_path),
                crop_path=str(crop_path),
                bbox=face.bbox,
                confidence=face.confidence,
                blur_score=report.blur_score,
                face_size_px=report.face_size_px,
            ))

    batch.elapsed_seconds = time.time() - start
    batch.detector_stats = detector.stats()
    detector.close()

    log.info(
        "face_extraction_complete",
        crops_saved=batch.count,
        rejected=batch.rejected,
        no_face_frames=batch.frames_with_no_face,
        elapsed_s=round(batch.elapsed_seconds, 2),
        **batch.detector_stats,
    )

    # Write face manifest for Day 3 to pick up
    face_manifest_path = crops_dir / "face_manifest.json"
    face_manifest = {
        "video_id": video_id,
        "crops_dir": str(crops_dir),
        "crop_count": batch.count,
        "output_size": output_size,
        "records": [asdict(r) for r in batch.records],
        "detector_stats": batch.detector_stats,
        "source_manifest": str(manifest_path),
    }
    face_manifest_path.write_text(json.dumps(face_manifest, indent=2))
    log.info("face_manifest_written", path=str(face_manifest_path))

    return batch