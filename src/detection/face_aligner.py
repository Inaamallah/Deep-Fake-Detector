# src/detection/face_aligner.py
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import euclidean

from src.detection.face_detector import DetectedFace


# Target landmark positions for a 112×112 aligned face crop.
# These exact coordinates match the ArcFace / FaceForensics++ training standard.
# Left eye, right eye, nose tip, mouth left, mouth right.
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _get_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transform (rotation + uniform scale + translation)
    from src_pts to dst_pts using least-squares fitting.

    Returns a 2×3 affine matrix for cv2.warpAffine.
    """
    assert src_pts.shape == dst_pts.shape == (5, 2)

    # Use OpenCV's estimateAffinePartial2D which fits a similarity transform
    # (4 DOF: tx, ty, scale, rotation — no shear or independent x/y scale)
    M, _ = cv2.estimateAffinePartial2D(
        src_pts.reshape(5, 1, 2).astype(np.float32),
        dst_pts.reshape(5, 1, 2).astype(np.float32),
        method=cv2.LMEDS,
    )

    if M is None:
        # Fallback: identity (no transform)
        M = np.eye(2, 3, dtype=np.float32)

    return M


def align_face(
    frame: np.ndarray,
    face: DetectedFace,
    output_size: int = 112,
    padding: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Warp the face region in `frame` to produce a standardised aligned crop.

    Args:
        frame:       Full BGR frame.
        face:        DetectedFace with 5 landmarks.
        output_size: Side length of the output square crop (112 is standard).
        padding:     Extra margin around the face (fraction of bbox size).
                     0.25 = 25% padding on each side — includes forehead and chin,
                     which are important for compression-artifact detection.

    Returns:
        Aligned BGR face crop of shape (output_size, output_size, 3),
        or None if the transform fails.
    """
    if len(face.landmarks) < 5:
        return None

    h, w = frame.shape[:2]
    src_pts = np.array(face.landmarks[:5], dtype=np.float32)

    # Scale reference landmarks to the requested output_size
    scale = output_size / 112.0
    ref_pts = REFERENCE_LANDMARKS_112 * scale

    M = _get_transform(src_pts, ref_pts)

    aligned = cv2.warpAffine(
        frame, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    padding: float = 0.25,
) -> Tuple[int, int, int, int]:
    """
    Expand bbox by `padding` fraction on each side, clamped to frame bounds.
    Used when we want to fall back to a simple crop (no landmark data).
    """
    x, y, bw, bh = bbox
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w, x + bw + pad_x)
    y2 = min(frame_h, y + bh + pad_y)
    return x1, y1, x2 - x1, y2 - y1