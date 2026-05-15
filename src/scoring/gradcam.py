# src/scoring/gradcam.py  ← NEW FILE
"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNet-B4.

Grad-CAM answers the question: "which spatial regions of this face caused
the model to predict it is a deepfake?"

Algorithm:
  1. Register forward and backward hooks on the target convolutional layer.
  2. Run a forward pass — the hook stores the layer's activation tensor.
  3. Run a backward pass — the hook stores the gradients flowing back.
  4. Compute per-channel importance weights by globally average-pooling
     the gradients: w_c = (1/HW) * Σ_{i,j} (∂score/∂A^c_{ij})
  5. Weight each activation channel by its importance: Σ_c w_c * A^c
  6. Apply ReLU to keep only regions that *increase* the fake score.
  7. Resize to input resolution and overlay on the original face crop.

The target layer is `conv_head` — the last 1×1 convolution in EfficientNet-B4
before global average pooling. Its output is spatially 14×14 for a 224×224
input, capturing high-level semantic regions with spatial specificity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import settings
from src.detection.model_downloader import (
    FINETUNED_WEIGHTS_PATH,
    build_pytorch_model,
)
from src.detection.preprocessor import preprocess_face
from src.utils.logger import logger


@dataclass
class HeatmapResult:
    """Paths and metadata for one generated heatmap."""
    frame_idx: int
    prob_fake: float
    crop_path: str
    heatmap_path: str   # raw JET colourmap on CAM
    overlay_path: str   # 50/50 blend of heatmap and original face


class GradCAM:
    """
    Thin wrapper that attaches hooks to a model and generates CAMs on demand.

    Lifecycle:
        cam = GradCAM(model)
        for img_tensor in tensors:
            cam_map = cam.generate(img_tensor)
        cam.remove_hooks()   # always call this when done

    Thread safety: NOT thread-safe. Use one GradCAM instance per thread.
    """

    TARGET_LAYER = "conv_head"

    def __init__(self, model: nn.Module) -> None:
        self.model        = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks:       list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        named = dict(self.model.named_modules())

        if self.TARGET_LAYER not in named:
            available = [n for n in named if n]
            raise ValueError(
                f"Layer '{self.TARGET_LAYER}' not found in model. "
                f"Last 5 named layers: {available[-5:]}"
            )

        layer = named[self.TARGET_LAYER]

        def _fwd(module: nn.Module, inp, output: torch.Tensor) -> None:
            # .clone() prevents the stored tensor from being overwritten
            # if PyTorch reuses the buffer in a subsequent forward pass.
            self._activations = output.detach().clone()

        def _bwd(module: nn.Module, grad_in, grad_out) -> None:
            # grad_out is a tuple; index 0 is the gradient w.r.t. the output.
            self._gradients = grad_out[0].detach().clone()

        self._hooks.append(layer.register_forward_hook(_fwd))
        self._hooks.append(layer.register_full_backward_hook(_bwd))

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a spatial activation map for one preprocessed image.

        Args:
            img_tensor: shape (1, 3, 224, 224), float32, ImageNet-normalised.
                        Must NOT be inside a torch.no_grad() context.

        Returns:
            cam: shape (H_feat, W_feat), float32, values in [0, 1].
                 Resize this to your desired output resolution with cv2.resize.
        """
        # Reset stored tensors from any previous call
        self._activations = None
        self._gradients   = None

        # Forward pass — hooks fire here
        output = self.model(img_tensor)   # shape: (1, 1)

        # Backward pass — hooks fire here
        self.model.zero_grad()
        output.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "GradCAM hooks did not capture activations/gradients. "
                "Ensure the model is not inside torch.no_grad()."
            )

        # _activations: (1, C, H, W)
        # _gradients:   (1, C, H, W)

        # Importance weight for each channel = spatial mean of its gradient
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum over channels
        cam = (weights * self._activations).sum(dim=1)               # (1, H, W)
        cam = torch.relu(cam)                                         # keep positives
        cam = cam.squeeze(0).numpy()                                  # (H, W)

        # Normalise to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()


def _load_model_for_gradcam() -> nn.Module:
    """
    Load the model for Grad-CAM inference.

    Priority: fine-tuned weights > ImageNet pretrained.
    We always use pretrained=False + explicit load_state_dict when
    fine-tuned weights exist, because that gives us the exact parameters
    the model converged to during fine-tuning.
    """
    log = logger.bind(component="gradcam_loader")

    if FINETUNED_WEIGHTS_PATH.exists():
        log.info("gradcam_loading_finetuned", path=str(FINETUNED_WEIGHTS_PATH))
        model = build_pytorch_model(pretrained=False)
        state_dict = torch.load(
            str(FINETUNED_WEIGHTS_PATH), map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        log.warning(
            "finetuned_weights_not_found_gradcam_will_use_imagenet",
            note="Heatmaps will not reflect deepfake-specific activations.",
        )
        model = build_pytorch_model(pretrained=True)

    # eval() is essential: it disables Dropout and switches BatchNorm to
    # use running statistics. Grad-CAM needs deterministic forward passes.
    model.eval()
    return model


class GradCAMGenerator:
    """
    Orchestrates Grad-CAM generation for a list of face crops.

    Usage:
        gen = GradCAMGenerator(output_dir=Path("data/results/abc/heatmaps"))
        results = gen.generate_for_crops(crop_paths, frame_indices, prob_fakes)
        gen.close()
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger.bind(component="gradcam_generator")

        self._model   = _load_model_for_gradcam()
        self._gradcam = GradCAM(self._model)

    def generate_for_crops(
        self,
        crop_paths:    List[str],
        frame_indices: List[int],
        prob_fakes:    List[float],
    ) -> List[HeatmapResult]:
        """
        Generate one heatmap + overlay per crop.

        Args:
            crop_paths:    File paths to the aligned face PNG files.
            frame_indices: Frame index for each crop (for filename labelling).
            prob_fakes:    P(fake) score for each crop (stored in the result).

        Returns:
            List of HeatmapResult, one per successfully processed crop.
            Crops that fail (unreadable file, hook error) are logged and skipped.
        """
        results: List[HeatmapResult] = []

        for crop_path_str, frame_idx, prob_fake in zip(
            crop_paths, frame_indices, prob_fakes
        ):
            crop_path = Path(crop_path_str)
            if not crop_path.exists():
                self.log.warning("crop_missing", path=str(crop_path))
                continue

            img = cv2.imread(str(crop_path))
            if img is None:
                self.log.warning("crop_unreadable", path=str(crop_path))
                continue

            try:
                result = self._process_one_crop(img, crop_path, frame_idx, prob_fake)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                self.log.warning(
                    "heatmap_failed", frame_idx=frame_idx, error=str(exc)
                )

        self.log.info("heatmaps_done", generated=len(results))
        return results

    def _process_one_crop(
        self,
        img:       np.ndarray,
        crop_path: Path,
        frame_idx: int,
        prob_fake: float,
    ) -> Optional[HeatmapResult]:
        # Preprocess to model input
        img_np = preprocess_face(img)                         # (1,3,224,224) float32
        img_t  = torch.from_numpy(img_np.copy())              # copy for contiguous memory

        # Generate CAM — this runs a forward + backward pass
        cam = self._gradcam.generate(img_t)                   # (H_feat, W_feat)

        # Resize CAM to match the original crop's pixel dimensions
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert float [0,1] CAM to a JET colour heatmap (BGR)
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # Overlay: addWeighted blends original face (50%) and heatmap (50%)
        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        # Save both artefacts
        stem         = f"frame_{frame_idx:06d}"
        heatmap_path = self.output_dir / f"{stem}_heatmap.png"
        overlay_path = self.output_dir / f"{stem}_overlay.png"

        cv2.imwrite(str(heatmap_path), heatmap)
        cv2.imwrite(str(overlay_path), overlay)

        return HeatmapResult(
            frame_idx    = frame_idx,
            prob_fake    = prob_fake,
            crop_path    = str(crop_path),
            heatmap_path = str(heatmap_path),
            overlay_path = str(overlay_path),
        )

    def close(self) -> None:
        """Release hooks. Always call this when finished."""
        self._gradcam.remove_hooks()