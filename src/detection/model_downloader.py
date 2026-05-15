# src/detection/model_downloader.py  ← FULL REPLACEMENT
from __future__ import annotations

from pathlib import Path

from config import settings
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Canonical file paths — imported by trainer.py and inference_engine.py.
# Keeping them here in one place means changing a path never requires hunting
# through multiple files.
# ---------------------------------------------------------------------------
ONNX_MODEL_PATH        = settings.models_dir / "deepfake_efficientb4.onnx"
FINETUNED_WEIGHTS_PATH = settings.models_dir / "deepfake_efficientb4_finetuned.pt"

# These must stay identical to the values in preprocessor.py.
# They represent the channel-wise mean and std of ImageNet, which is what
# the EfficientNet-B4 backbone was originally trained on.
INPUT_SIZE = 224
MEAN       = (0.485, 0.456, 0.406)
STD        = (0.229, 0.224, 0.225)


def build_pytorch_model(pretrained: bool = True):
    """
    Build EfficientNet-B4 with a single-logit binary output head.

    Args:
        pretrained: When True, timm downloads and loads ImageNet weights for
                    the backbone. These weights already "understand" edges,
                    textures, and shapes — a huge head start for fine-tuning.
                    When False, the architecture is returned with random weights.
                    Only pass False when you are about to overwrite the weights
                    immediately with load_state_dict(), as export_to_onnx() does.

    Returns:
        A timm EfficientNet-B4 nn.Module with num_classes=1.
    """
    import timm

    model = timm.create_model(
        "efficientnet_b4",
        pretrained=pretrained,  # True = ImageNet knowledge included
        num_classes=1,          # single logit → apply sigmoid → P(fake)
        drop_rate=0.4,          # dropout is automatically disabled in model.eval()
    )
    return model


def export_to_onnx(force: bool = False) -> Path:
    """
    Convert model weights to ONNX format for CPU inference.

    Loading priority (first match wins):
      1. Fine-tuned weights at FINETUNED_WEIGHTS_PATH
         Produced by trainer.py. Specialised for deepfake detection.
      2. ImageNet pretrained weights (downloaded by timm automatically)
         A safe fallback for pipeline testing before fine-tuning is done.
         Will not be accurate for deepfake detection specifically.

    Args:
        force: Re-export even if the ONNX file already exists.

    Returns:
        Path to the ONNX file.
    """
    import torch

    log = logger.bind(component="onnx_export")

    if ONNX_MODEL_PATH.exists() and not force:
        log.info("onnx_already_exists", path=str(ONNX_MODEL_PATH))
        return ONNX_MODEL_PATH

    # --- Decide which weights to export ---
    if FINETUNED_WEIGHTS_PATH.exists():
        log.info("loading_finetuned_weights", path=str(FINETUNED_WEIGHTS_PATH))

        # pretrained=False here because we are about to replace every weight
        # immediately with load_state_dict(). Downloading ImageNet weights first
        # would be wasted time and bandwidth.
        model = build_pytorch_model(pretrained=False)

        state_dict = torch.load(
            str(FINETUNED_WEIGHTS_PATH), map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=True)
        log.info("finetuned_weights_loaded_successfully")

    else:
        log.warning(
            "finetuned_weights_not_found_using_imagenet_pretrained",
            expected=str(FINETUNED_WEIGHTS_PATH),
            action="Run 'python cli.py finetune ...' first for real accuracy.",
        )
        # ImageNet weights give you a working pipeline to test against,
        # but the model has not learned deepfake-specific artifacts yet.
        model = build_pytorch_model(pretrained=True)

    # Switch to eval mode: disables dropout, sets BatchNorm to use
    # running statistics instead of batch statistics.
    model.eval()

    log.info(
        "exporting_to_onnx",
        input_size=INPUT_SIZE,
        output=str(ONNX_MODEL_PATH),
    )

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            # Batch size is dynamic so the same ONNX model handles
            # both single-face and batched inference without re-exporting.
            "pixel_values": {0: "batch_size"},
            "logits":       {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,  # fuses constant subgraphs → faster CPU inference
    )

    log.info("onnx_export_complete", path=str(ONNX_MODEL_PATH))
    return ONNX_MODEL_PATH