# src/detection/inference_engine.py
"""
ONNX Runtime inference engine for deepfake detection.

Design principles:
  1. Load the ONNX session once, reuse it across many calls.
  2. Process crops in batches to maximise CPU utilisation.
  3. Apply temperature scaling to produce well-calibrated probabilities.
  4. Return typed dataclasses, never raw numpy arrays, so downstream
     code doesn't have to guess about shapes and dtypes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from config import settings
from src.detection.model_downloader import ONNX_MODEL_PATH, export_to_onnx
from src.detection.preprocessor import preprocess_batch
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
# Neural networks are famously over-confident — a model might output a
# logit of 4.5 (sigmoid ≈ 0.99) when the true underlying probability is
# closer to 0.75. Temperature scaling divides the raw logit by a learned
# scalar T before applying sigmoid. T > 1 softens (flattens) the distribution;
# T < 1 sharpens it. The value below was calibrated on a held-out validation
# split of FaceForensics++. If you re-train or fine-tune, recalibrate T.
TEMPERATURE = 1.5


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FaceScore:
    """Inference result for a single face crop."""
    crop_path: str       # path to the face PNG file
    frame_idx: int
    face_idx: int
    logit: float         # raw model output (pre-sigmoid)
    prob_fake: float     # P(fake) after temperature scaling + sigmoid
    is_fake: bool        # hard decision at the configured threshold
    latency_ms: float    # inference time for this face


@dataclass
class VideoInferenceResult:
    """Aggregated inference result for a whole video."""
    video_id: str
    face_scores: List[FaceScore]
    total_faces_scored: int
    mean_prob_fake: float        # simple average across all faces
    weighted_prob_fake: float    # confidence-weighted average (see below)
    verdict: str                 # "REAL" or "DEEPFAKE"
    overall_confidence: float    # how confident we are in the verdict (0–1)
    fake_frame_ratio: float      # fraction of scored faces classified as fake
    decision_threshold: float    # video-level threshold used for the verdict
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------
class DeepfakeInferenceEngine:
    """
    Loads the ONNX model and exposes a clean API for running inference.

    Thread safety: ONNX Runtime sessions are thread-safe for concurrent
    infer() calls. This engine is safe to share across threads.
    """

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        batch_size: int = 8,
        temperature: float = TEMPERATURE,
        num_threads: int = 4,
    ):
        """
        Args:
            onnx_path:    Path to the .onnx file. Auto-exports if not found.
            batch_size:   Faces processed per ONNX call. 8 is a good default
                          for 4-core CPUs. Increase if you have more cores.
            temperature:  Calibration temperature (T). Higher = more uncertain.
            num_threads:  ONNX Runtime inter-op thread count. Match to your
                          machine's physical cores for best throughput.
        """
        self.batch_size  = batch_size
        self.temperature = temperature
        self.log         = logger.bind(component="inference_engine")

        # Ensure ONNX model exists — convert if needed
        if onnx_path is None:
            onnx_path = ONNX_MODEL_PATH
        if not onnx_path.exists():
            self.log.info("onnx_not_found_exporting")
            onnx_path = export_to_onnx()

        # Configure ONNX Runtime session options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads   # parallelise within ops
        opts.inter_op_num_threads = num_threads   # parallelise across ops
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Enables: constant folding, common subexpression elimination,
            # node fusion (e.g. Conv+BN+ReLU fused into one kernel).
            # This is where most of the CPU speed-up comes from.
        )
        opts.enable_mem_pattern = True   # reuse memory allocations

        # CPU execution provider — the only one we need (no GPU)
        providers = [("CPUExecutionProvider", {})]

        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=providers
        )

        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self.log.info(
            "engine_ready",
            onnx_path=str(onnx_path),
            batch_size=batch_size,
            temperature=temperature,
            input_name=self._input_name,
        )

    # -----------------------------------------------------------------------
    # Core inference logic
    # -----------------------------------------------------------------------
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: avoids overflow for large |x| values."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    def _run_batch(
    self,
    crops: List[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run ONNX inference on a list of face crops.
        Returns an array of P(fake) values, one per crop.

        The temperature scaling step is:
            prob = sigmoid(logit / T)
        Dividing by T > 1 brings extreme logits (e.g. ±5) closer to 0,
        which produces more honest uncertainty estimates.
        """
        batch = preprocess_batch(crops)              # (N, 3, 224, 224)
        logits = self._session.run(
            [self._output_name], {self._input_name: batch}
        )[0]                                         # shape: (N, 1) or (N,)

        logits = logits.squeeze()                    # → (N,) or scalar
        if logits.ndim == 0:
            logits = logits[np.newaxis]              # handle N=1 edge case

        # Apply temperature scaling then sigmoid
        # Apply temperature scaling then sigmoid
        scaled_logits = logits / self.temperature
        probs = self._sigmoid(scaled_logits)

        # Return BOTH logits and probabilities
        return logits.astype(np.float32), probs.astype(np.float32)

    def score_crop_list(
        self,
        crop_paths: List[Path],
        frame_indices: List[int],
        face_indices: List[int],
        threshold: float = 0.5,
    ) -> List[FaceScore]:
        """
        Score a list of face crop images from disk.

        Processes them in batches of self.batch_size for efficiency.
        Returns one FaceScore per crop, in the same order as the input.
        """
        assert len(crop_paths) == len(frame_indices) == len(face_indices)

        self.log.info("scoring_crops", count=len(crop_paths))
        scores: List[FaceScore] = []

        # Process in batches
        for batch_start in tqdm(
            range(0, len(crop_paths), self.batch_size),
            desc="Running inference",
            unit="batch",
            leave=False,
        ):
            batch_end = min(batch_start + self.batch_size, len(crop_paths))
            batch_paths = crop_paths[batch_start:batch_end]

            # Load crops from disk
            crops = []
            valid_indices = []  # track which crops loaded successfully
            for i, path in enumerate(batch_paths):
                img = cv2.imread(str(path))
                if img is None:
                    self.log.warning("crop_unreadable", path=str(path))
                    continue
                crops.append(img)
                valid_indices.append(batch_start + i)

            if not crops:
                continue

            # Time the inference call
            t0 = time.perf_counter()

            logits, probs = self._run_batch(crops)

            latency_ms = (time.perf_counter() - t0) * 1000 / len(crops)

            for i, (logit, prob, global_idx) in enumerate(
                zip(logits, probs, valid_indices)
            ):
                scores.append(FaceScore(
                    crop_path=str(crop_paths[global_idx]),
                    frame_idx=frame_indices[global_idx],
                    face_idx=face_indices[global_idx],
                    logit=float(logit),
                    prob_fake=float(prob),
                    is_fake=bool(prob >= threshold),
                    latency_ms=round(latency_ms, 2),
                ))

        self.log.info(
            "scoring_complete",
            scored=len(scores),
            mean_prob=round(float(np.mean([s.prob_fake for s in scores])), 3)
            if scores else 0,
        )
        return scores
