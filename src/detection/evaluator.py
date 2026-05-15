"""
Evaluate fine-tuned model on train/val/test sets with confusion matrices.
"""
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.detection.dataset import FaceForensicsDataset
from src.detection.model_downloader import (
    FINETUNED_WEIGHTS_PATH,
    build_pytorch_model,
)
from src.detection.inference_engine import DeepfakeInferenceEngine
from src.detection.video_scorer import score_video_from_manifest
from src.utils.logger import logger


def evaluate_model(
    real_manifest_paths: List[Path],
    fake_manifest_paths: List[Path],
    weights_path: Path = FINETUNED_WEIGHTS_PATH,
    batch_size: int = 16,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 0,
    output_dir: Path = None,
) -> dict:
    """
    Evaluate fine-tuned model on train/val/test splits with metrics and confusion matrices.

    Args:
        real_manifest_paths: Paths to real face_manifest.json files.
        fake_manifest_paths: Paths to fake face_manifest.json files.
        weights_path: Path to saved model weights (.pt file).
        batch_size: Batch size for evaluation.
        val_split: Fraction of data for validation.
        test_split: Fraction of remaining data for testing.
        num_workers: DataLoader workers.
        output_dir: Directory to save confusion matrix plots. If None, skips saving.

    Returns:
        Dictionary with metrics for each split:
        {
            "train": {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "auc": ...},
            "val": {...},
            "test": {...},
            "confusion_matrices": {
                "train": [...],
                "val": [...],
                "test": [...]
            }
        }
    """
    log = logger.bind(component="evaluator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("evaluation_start", device=str(device))

    # Load dataset
    dataset = FaceForensicsDataset(
        real_manifest_paths=real_manifest_paths,
        fake_manifest_paths=fake_manifest_paths,
        is_training=False,  # no augmentations for evaluation
    )

    total = len(dataset)
    log.info("dataset_loaded", total_samples=total)

    # Create deterministic splits (train/val/test)
    generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(total, generator=generator).tolist()

    n_val = max(1, int(total * val_split))
    n_test = max(1, int((total - n_val) * test_split))
    n_train = total - n_val - n_test

    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:n_train + n_val]
    test_indices = all_indices[n_train + n_val:]

    log.info("data_split", train=n_train, val=n_val, test=n_test)

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Load model
    model = build_pytorch_model(pretrained=False)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        log.info("weights_loaded", path=str(weights_path))
    else:
        log.warning("weights_not_found", path=str(weights_path))

    model = model.to(device)
    model.eval()

    # Evaluate on each split
    results = {"confusion_matrices": {}}

    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        log.info(f"evaluating_{split_name}")
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                labels_np = labels.cpu().numpy().astype(int)

                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                all_labels.extend(labels_np.tolist())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        if len(set(all_labels)) >= 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5

        cm = confusion_matrix(all_labels, all_preds)
        results["confusion_matrices"][split_name] = cm.tolist()

        results[split_name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "total_samples": int(len(all_labels)),
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        }

        log.info(
            f"{split_name}_metrics",
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            auc=round(auc, 4),
        )

    # Save confusion matrix plots if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "val", "test"]:
            cm = np.array(results["confusion_matrices"][split_name])
            _plot_confusion_matrix(
                cm, split_name,
                output_path=output_dir / f"confusion_matrix_{split_name}.png"
            )
            log.info(f"confusion_matrix_saved", split=split_name,
                     path=str(output_dir / f"confusion_matrix_{split_name}.png"))

    return results


def _plot_confusion_matrix(cm: np.ndarray, split_name: str, output_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix — {split_name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()


def calibrate_video_threshold(
    real_manifest_paths: List[Path],
    fake_manifest_paths: List[Path],
    batch_size: int = 8,
    thresholds: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Score labelled videos and choose a better video-level fake threshold.

    This evaluates whole videos, not random face crops. That matters because
    production decisions are made per video, and crop-level random splits can
    leak near-duplicate frames from one video into both train and validation.

    Returns a dict with per-video scores, all threshold metrics, and the best
    threshold selected by F1 score with balanced accuracy as a tie-breaker.
    """
    log = logger.bind(component="video_threshold_calibration")

    labelled_manifests = (
        [(Path(p), 0) for p in real_manifest_paths]
        + [(Path(p), 1) for p in fake_manifest_paths]
    )
    if len(labelled_manifests) < 2:
        raise ValueError("Need at least one real and one fake manifest.")

    if not real_manifest_paths or not fake_manifest_paths:
        raise ValueError("Calibration requires both real and fake videos.")

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    engine = DeepfakeInferenceEngine(batch_size=batch_size)
    video_rows = []

    for manifest_path, label in labelled_manifests:
        result = score_video_from_manifest(
            manifest_path,
            engine=engine,
            batch_size=batch_size,
            threshold=0.5,
        )
        video_rows.append({
            "video_id": result.video_id,
            "label": int(label),
            "label_name": "fake" if label else "real",
            "weighted_prob_fake": float(result.weighted_prob_fake),
            "mean_prob_fake": float(result.mean_prob_fake),
            "fake_frame_ratio": float(result.fake_frame_ratio),
            "faces_scored": int(result.total_faces_scored),
            "manifest_path": str(manifest_path),
        })

    y_true = np.array([row["label"] for row in video_rows], dtype=np.int32)
    y_score = np.array([row["weighted_prob_fake"] for row in video_rows], dtype=np.float32)

    threshold_rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.int32)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        threshold_rows.append({
            "threshold": round(float(threshold), 4),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        })

    best = max(
        threshold_rows,
        key=lambda row: (
            row["f1"],
            row["balanced_accuracy"],
            row["recall"],
            -abs(row["threshold"] - 0.5),
        ),
    )

    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) == 2 else 0.5
    result = {
        "recommended_threshold": best["threshold"],
        "selection_metric": "max_f1_then_balanced_accuracy",
        "auc": auc,
        "best_metrics": best,
        "videos": video_rows,
        "thresholds": threshold_rows,
    }

    log.info(
        "threshold_calibrated",
        videos=len(video_rows),
        recommended_threshold=best["threshold"],
        f1=round(best["f1"], 4),
        balanced_accuracy=round(best["balanced_accuracy"], 4),
        auc=round(auc, 4),
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "threshold_calibration.json").write_text(
            json.dumps(result, indent=2)
        )
        _write_calibration_csv(video_rows, output_dir / "video_scores.csv")
        _plot_score_distribution(video_rows, best["threshold"], output_dir / "score_distribution.png")

    return result


def _write_calibration_csv(rows: List[dict], output_path: Path) -> None:
    header = [
        "video_id", "label", "label_name", "weighted_prob_fake",
        "mean_prob_fake", "fake_frame_ratio", "faces_scored", "manifest_path",
    ]
    lines = [",".join(header)]
    for row in rows:
        values = [str(row.get(key, "")) for key in header]
        lines.append(",".join(v.replace(",", " ") for v in values))
    output_path.write_text("\n".join(lines) + "\n")


def _plot_score_distribution(rows: List[dict], threshold: float, output_path: Path) -> None:
    real_scores = [r["weighted_prob_fake"] for r in rows if r["label"] == 0]
    fake_scores = [r["weighted_prob_fake"] for r in rows if r["label"] == 1]

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)
    plt.hist(real_scores, bins=bins, alpha=0.7, label="Real", color="#2E86AB")
    plt.hist(fake_scores, bins=bins, alpha=0.7, label="Fake", color="#D1495B")
    plt.axvline(threshold, color="#222222", linestyle="--", label=f"Threshold {threshold:.2f}")
    plt.xlabel("Weighted P(fake)")
    plt.ylabel("Video count")
    plt.title("Video Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close()
