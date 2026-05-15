# src/detection/dataset.py  ← NEW FILE, create it
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class PILToTensorNoNumpy:
    """Convert a PIL RGB image to a CHW float tensor without torch.from_numpy."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        tensor = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(height, width, 3)
        return tensor.permute(2, 0, 1).contiguous().float().div(255.0)


class FaceForensicsDataset(Dataset):
    """
    Loads aligned face crops from face_manifest.json files produced by Day 2.

    Each sample is a (image_tensor, label) pair where:
        label = 0  →  authentic (real) video
        label = 1  →  deepfake video

    In training mode the dataset applies mild augmentations to improve
    generalisation. In validation mode it applies only the normalisation
    required by the model, so validation metrics reflect true performance.
    """

    # These must match preprocessor.py exactly. The model's expectations
    # are baked in at training time; deviating here in inference would
    # silently corrupt predictions.
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        real_manifest_paths: List[Path],
        fake_manifest_paths: List[Path],
        is_training: bool = True,
        input_size: int = 224,
        max_crops_per_video: int = 50,
    ):
        """
        Args:
            real_manifest_paths:   Paths to face_manifest.json files for real videos.
            fake_manifest_paths:   Paths to face_manifest.json files for fake videos.
            is_training:           True → apply augmentations during __getitem__.
            input_size:            Target spatial size for the model input (224 for EfficientNet-B4).
            max_crops_per_video:   Maximum crops to take from a single video.
                                   A 10-minute video might have 500 crops; without this cap
                                   it would dominate the dataset and introduce temporal bias.
        """
        self.is_training = is_training
        self.input_size  = input_size

        # Flat list of (absolute_crop_path, label) tuples.
        # We keep this as plain strings rather than Path objects because
        # Python's multiprocessing serialises strings faster than Path objects
        # when num_workers > 0 in the DataLoader.
        self.samples: List[Tuple[str, int]] = []

        self._load_manifests(real_manifest_paths, label=0, max_crops=max_crops_per_video)
        self._load_manifests(fake_manifest_paths, label=1, max_crops=max_crops_per_video)

        # Build the transform once here rather than inside __getitem__.
        # Building a Compose object is not free; doing it per-sample would
        # add meaningful overhead with large datasets.
        self._transform = self._build_transform()

    def _load_manifests(
        self,
        manifest_paths: List[Path],
        label: int,
        max_crops: int,
    ) -> None:
        """Read crop file paths from each manifest and append them to self.samples."""
        for manifest_path in manifest_paths:
            if not manifest_path.exists():
                continue

            manifest = json.loads(manifest_path.read_text())
            records  = manifest.get("records", [])

            # We take the first max_crops records in the order they appear in
            # the manifest. We do NOT shuffle here — the DataLoader's shuffle=True
            # handles randomisation during training, and keeping the dataset
            # ordering deterministic makes the train/val split reproducible.
            for record in records[:max_crops]:
                crop_path = record.get("crop_path", "")
                if crop_path and Path(crop_path).exists():
                    self.samples.append((crop_path, label))

    def _build_transform(self) -> T.Compose:
        """
        Construct the torchvision preprocessing pipeline.

        Training augmentations are deliberately conservative because deepfake
        artifacts are subtle. Aggressive augmentations like large crops or
        strong blur can destroy the exact pixel patterns the model needs to
        learn. We use:
          - Horizontal flip: faces are roughly symmetric, so this is free data.
          - Mild colour jitter: mimics lighting variation across cameras.
          - Small rotation (±10°): covers natural head-tilt variation.
        """
        if self.is_training:
            return T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.05,
                ),
                T.RandomRotation(degrees=10),
                T.Resize((self.input_size, self.input_size)),
                PILToTensorNoNumpy(),
                T.Normalize(mean=self.MEAN, std=self.STD),
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.Resize((self.input_size, self.input_size)),
                PILToTensorNoNumpy(),
                T.Normalize(mean=self.MEAN, std=self.STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        crop_path, label = self.samples[idx]

        img = cv2.imread(crop_path)
        if img is None:
            # Return a black image rather than raising — this lets the
            # DataLoader continue cleanly if a file was deleted after the
            # manifest was written. The model will score it near 0.5 (uncertain),
            # which is the safest possible outcome for a corrupted sample.
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        # OpenCV loads as BGR; torchvision Normalize and the model expect RGB.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor   = self._transform(img_rgb)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor

    def class_balance(self) -> dict:
        """Return real/fake sample counts, useful for pos_weight calculation."""
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        total  = len(labels)
        return {
            "real":       n_real,
            "fake":       n_fake,
            "total":      total,
            "real_ratio": round(n_real / max(total, 1), 3),
            "fake_ratio": round(n_fake / max(total, 1), 3),
        }
