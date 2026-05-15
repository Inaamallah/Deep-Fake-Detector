# src/detection/preprocessor.py
"""
Image preprocessing for deepfake inference.

The normalisation constants (mean, std) and the resize target (224×224)
MUST match what was used when the EfficientNet-B4 was fine-tuned on FF++.
Changing even one of these silently destroys accuracy, which is why they
live here rather than scattered across the codebase.
"""
from __future__ import annotations

import cv2
import numpy as np

# These are standard ImageNet constants — the FF++ fine-tune used them too.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The model's expected spatial input — fixed at export time.
MODEL_INPUT_SIZE = 224


def preprocess_face(crop: np.ndarray) -> np.ndarray:
    """
    Prepare one aligned face crop (BGR, uint8) for model inference.

    The transformation pipeline is:
      BGR → RGB              (OpenCV loads BGR; the model expects RGB)
      Resize to 224×224      (model's fixed spatial requirement)
      uint8 [0,255] → float32 [0.0,1.0]
      Normalise: (x - mean) / std
      HWC → CHW              (PyTorch/ONNX convention: channels first)
      Add batch dim: CHW → NCHW  (shape becomes [1, 3, 224, 224])

    Returns an NCHW float32 numpy array ready to pass to ONNX Runtime.

    Mental model: think of normalisation as "centering" each colour
    channel around 0 and scaling its range. A pixel value of 0.485
    (which is the mean for the red channel) becomes 0.0 after
    normalisation. This zero-centering makes gradient descent during
    training faster and more stable — so we replicate it at inference.
    """
    # Step 1: colour space conversion
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Step 2: resize to model input size using bilinear interpolation.
    # We use INTER_LINEAR (bilinear) rather than INTER_AREA because the
    # crops are already close to 112px and we are upscaling to 224px.
    resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)

    # Step 3: float conversion and [0,1] scaling
    img = resized.astype(np.float32) / 255.0

    # Step 4: channel-wise normalisation
    img = (img - MEAN) / STD

    # Step 5: HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)          # (H, W, C) → (C, H, W)
    img = np.expand_dims(img, axis=0)     # (C, H, W) → (1, C, H, W)

    return img


def preprocess_batch(crops: list[np.ndarray]) -> np.ndarray:
    """
    Preprocess a list of face crops into a single batched NCHW tensor.

    Batching is more efficient than processing one face at a time because
    ONNX Runtime can parallelise operations across the batch dimension.
    A batch size of 8–16 is the sweet spot for CPU inference.

    Returns shape: (N, 3, 224, 224) where N = len(crops).
    """
    processed = [preprocess_face(c) for c in crops]  # each is (1,3,224,224)
    return np.concatenate(processed, axis=0)          # → (N,3,224,224)