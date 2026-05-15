# ADD these two classes at the bottom of tests/test_day3.py.
# Everything already in the file stays exactly as it is.

import json
import tempfile
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np
import torch
import pytest
import cv2

# The three classes below are new additions.

class TestFaceForensicsDataset:
    """Tests for src/detection/dataset.py"""

    def _write_manifest(self, directory: Path, n_crops: int, name: str) -> Path:
        """Helper: write a face_manifest.json with real synthetic PNG crops."""
        crops_dir = directory / name
        crops_dir.mkdir()

        records = []
        for i in range(n_crops):
            crop_path = crops_dir / f"face_{i:04d}.png"
            # A 112×112 random image — passes quality checks, loads correctly.
            img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            cv2.imwrite(str(crop_path), img)
            records.append({
                "frame_idx":   i,
                "face_idx":    0,
                "crop_path":   str(crop_path),
                "bbox":        [10, 10, 100, 100],
                "confidence":  0.95,
                "blur_score":  120.0,
                "face_size_px": 100,
            })

        manifest = {
            "video_id":   f"test_{name}",
            "crops_dir":  str(crops_dir),
            "crop_count": n_crops,
            "output_size": 112,
            "records":    records,
        }
        manifest_path = directory / f"{name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_total_sample_count(self, tmp_path):
        from src.detection.dataset import FaceForensicsDataset
        real_m = self._write_manifest(tmp_path, n_crops=10, name="real_a")
        fake_m = self._write_manifest(tmp_path, n_crops=10, name="fake_a")
        ds = FaceForensicsDataset([real_m], [fake_m], is_training=False)
        assert len(ds) == 20

    def test_output_tensor_shape(self, tmp_path):
        from src.detection.dataset import FaceForensicsDataset
        real_m = self._write_manifest(tmp_path, n_crops=5, name="real_b")
        fake_m = self._write_manifest(tmp_path, n_crops=5, name="fake_b")
        ds = FaceForensicsDataset([real_m], [fake_m], is_training=False, input_size=224)
        img_t, lbl_t = ds[0]
        assert img_t.shape == (3, 224, 224)   # CHW format
        assert lbl_t.ndim == 0                # scalar tensor

    def test_label_dtype_is_float32(self, tmp_path):
        from src.detection.dataset import FaceForensicsDataset
        real_m = self._write_manifest(tmp_path, n_crops=3, name="real_c")
        fake_m = self._write_manifest(tmp_path, n_crops=3, name="fake_c")
        ds = FaceForensicsDataset([real_m], [fake_m], is_training=False)
        _, lbl_t = ds[0]
        # BCEWithLogitsLoss requires float32 labels, not int or long.
        assert lbl_t.dtype == torch.float32

    def test_both_labels_present(self, tmp_path):
        from src.detection.dataset import FaceForensicsDataset
        real_m = self._write_manifest(tmp_path, n_crops=5, name="real_d")
        fake_m = self._write_manifest(tmp_path, n_crops=5, name="fake_d")
        ds = FaceForensicsDataset([real_m], [fake_m], is_training=False)
        labels = {ds[i][1].item() for i in range(len(ds))}
        assert 0.0 in labels  # real
        assert 1.0 in labels  # fake

    def test_max_crops_cap(self, tmp_path):
        from src.detection.dataset import FaceForensicsDataset
        real_m = self._write_manifest(tmp_path, n_crops=20, name="real_e")
        fake_m = self._write_manifest(tmp_path, n_crops=20, name="fake_e")
        ds = FaceForensicsDataset([real_m], [fake_m], is_training=False, max_crops_per_video=5)
        # 5 real + 5 fake = 10, not 40
        assert len(ds) == 10

    def test_class_balance_correct(self, tmp_path):
        from src.detection.dataset import FaceForensicsDataset
        real_m = self._write_manifest(tmp_path, n_crops=8, name="real_f")
        fake_m = self._write_manifest(tmp_path, n_crops=4, name="fake_f")
        ds      = FaceForensicsDataset([real_m], [fake_m], is_training=False)
        balance = ds.class_balance()
        assert balance["real"]  == 8
        assert balance["fake"]  == 4
        assert balance["total"] == 12


class TestTrainerUtilities:
    """Tests for src/detection/trainer.py helper functions."""

    def test_freeze_early_layers_nonzero(self):
        from src.detection.trainer import freeze_early_layers
        from src.detection.model_downloader import build_pytorch_model
        model  = build_pytorch_model(pretrained=False)
        frozen = freeze_early_layers(model)
        total  = sum(p.numel() for p in model.parameters())
        # Some but not all parameters should be frozen
        assert 0 < frozen < total

    def test_blocks_0_and_1_are_frozen(self):
        from src.detection.trainer import freeze_early_layers
        from src.detection.model_downloader import build_pytorch_model
        model = build_pytorch_model(pretrained=False)
        freeze_early_layers(model)
        for name, param in model.named_parameters():
            if name.startswith("blocks.0.") or name.startswith("blocks.1."):
                assert not param.requires_grad, (
                    f"Parameter {name} should be frozen but requires_grad is True"
                )

    def test_classifier_head_remains_trainable(self):
        from src.detection.trainer import freeze_early_layers
        from src.detection.model_downloader import build_pytorch_model
        model = build_pytorch_model(pretrained=False)
        freeze_early_layers(model)
        for name, param in model.named_parameters():
            if "classifier" in name:
                assert param.requires_grad, (
                    f"Classifier parameter {name} should be trainable but is frozen"
                )

    def test_pos_weight_balanced_classes(self):
        from src.detection.trainer import compute_pos_weight
        mock_ds = MagicMock()
        mock_ds.class_balance.return_value = {
            "real": 100, "fake": 100, "total": 200
        }
        weight = compute_pos_weight(mock_ds)
        assert abs(weight.item() - 1.0) < 1e-5

    def test_pos_weight_imbalanced_classes(self):
        from src.detection.trainer import compute_pos_weight
        # 400 real, 100 fake → weight should be 4.0
        mock_ds = MagicMock()
        mock_ds.class_balance.return_value = {
            "real": 400, "fake": 100, "total": 500
        }
        weight = compute_pos_weight(mock_ds)
        assert abs(weight.item() - 4.0) < 1e-5

    def test_pos_weight_no_fake_samples(self):
        from src.detection.trainer import compute_pos_weight
        mock_ds = MagicMock()
        mock_ds.class_balance.return_value = {
            "real": 100, "fake": 0, "total": 100
        }
        # Should return 1.0 and not raise ZeroDivisionError
        weight = compute_pos_weight(mock_ds)
        assert weight.item() == 1.0