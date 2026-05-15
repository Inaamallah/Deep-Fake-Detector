# src/detection/trainer.py  ← NEW FILE, create it
from __future__ import annotations

import contextlib
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from src.detection.dataset import FaceForensicsDataset
from src.detection.model_downloader import (
    FINETUNED_WEIGHTS_PATH,
    build_pytorch_model,
)
from src.utils.logger import logger


@dataclass
class TrainingResult:
    weights_path: str
    best_val_auc: float
    best_val_acc: float
    best_epoch: int
    epochs_ran: int
    learning_rate: float
    batch_size: int
    weight_decay: float
    freeze_blocks: int
    patience: int


def freeze_early_layers(model: nn.Module, freeze_blocks: int = 2) -> int:
    """
    Freeze the first two MBConv block groups in EfficientNet-B4.

    The intuition here is important to understand. The earliest layers of a
    convolutional network are universal — they detect edges, colour gradients,
    and simple textures that look the same in real and fake images. There is
    nothing deepfake-specific to learn there, so letting them change during
    fine-tuning wastes computation and risks damaging well-learned general
    features with a small specialised dataset.

    The deeper blocks (2 through 6) detect higher-level patterns: facial
    structure, skin texture regularity, blending boundary sharpness. These are
    exactly where GAN and diffusion-based synthesis leaves traces, so those
    layers must remain trainable.

    Returns the number of frozen parameters so the caller can log it.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        # EfficientNet-B4 in timm names its MBConv groups as blocks.0,
        # blocks.1, blocks.2, ... blocks.6.
        if any(name.startswith(f"blocks.{i}.") for i in range(freeze_blocks)):
            param.requires_grad = False
            frozen_count += param.numel()
    return frozen_count


def compute_pos_weight(dataset: FaceForensicsDataset) -> torch.Tensor:
    """
    Compute the positive-class (fake) weight for BCEWithLogitsLoss.

    If your dataset has 800 real samples and 200 fake samples, an unweighted
    model will learn to always predict "real" — achieving 80% accuracy by
    never detecting a single deepfake. That is the class imbalance problem.

    pos_weight = n_real / n_fake tells the loss function to penalise a missed
    fake detection (false negative) four times more heavily than a missed real
    detection (false positive). This forces the model to take both classes
    seriously regardless of how many examples of each you have.

    Returns a scalar tensor suitable for passing directly to BCEWithLogitsLoss.
    """
    balance = dataset.class_balance()
    n_real  = balance["real"]
    n_fake  = balance["fake"]

    if n_fake == 0:
        # Degenerate case: no fake samples at all — weight is meaningless.
        return torch.tensor(1.0, dtype=torch.float32)

    weight = n_real / n_fake
    return torch.tensor(weight, dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    is_training: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Execute one full pass through the dataset.

    During training the function updates model weights via backpropagation.
    During validation it only runs forward passes inside torch.no_grad(),
    which skips building the autograd computation graph and saves ~40% of
    memory and time.

    Returns:
        (average_loss, accuracy, roc_auc) — all plain Python floats.

    Why AUC instead of just accuracy?
    AUC (Area Under the ROC Curve) measures the model's ability to rank a
    randomly chosen fake above a randomly chosen real, regardless of any
    threshold. A model that outputs 0.99 for all fakes and 0.01 for all reals
    has perfect AUC even if the 0.5 threshold gives mediocre accuracy.
    This makes it the right metric to track during training, especially with
    class imbalance.
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_probs:  List[float] = []
    all_labels: List[int]   = []
    correct = 0
    total   = 0

    # contextlib.nullcontext() is a no-op context manager — it does nothing
    # but satisfies the `with` block syntax when we are in training mode and
    # genuinely need gradients to flow.
    grad_ctx = contextlib.nullcontext() if is_training else torch.no_grad()

    with grad_ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # Forward pass. Model output shape is (batch, 1); squeeze to (batch,)
            # so it matches the (batch,) shape of labels for BCEWithLogitsLoss.
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping keeps the parameter updates bounded.
                # Without it, an unlucky batch early in training can produce
                # very large gradients that corrupt the weights catastrophically.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # Accumulate stats — detach from the graph before moving to CPU.
            total_loss += loss.item() * labels.size(0)

            probs     = torch.sigmoid(logits).detach().cpu().numpy()
            preds     = (probs >= 0.5).astype(int)
            labels_np = labels.detach().cpu().numpy().astype(int)

            correct += int((preds == labels_np).sum())
            total   += labels.size(0)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels_np.tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    # roc_auc_score raises ValueError if only one class is present in the batch,
    # which can happen with very small validation sets. We fall back to 0.5
    # (random-classifier AUC) in that edge case rather than crashing.
    if len(set(all_labels)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(all_labels, all_probs))

    return float(avg_loss), float(accuracy), auc


def fine_tune(
    real_manifest_paths: List[Path],
    fake_manifest_paths: List[Path],
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_blocks: int = 2,
    val_split: float = 0.2,
    patience: int = 3,
    num_workers: int = 0,
    output_path: Path = FINETUNED_WEIGHTS_PATH,
    save_metadata: bool = True,
) -> Path:
    """
    Fine-tune EfficientNet-B4 on your labelled face crop data.

    The complete process:
      1. Build two dataset objects (one with augmentations, one without)
         from the same manifest files.
      2. Split the indices deterministically so train and val use the exact
         same crop files, just accessed through different transform pipelines.
      3. Load EfficientNet-B4 with ImageNet pretrained weights.
      4. Freeze the first two block groups (generic feature detectors).
      5. Optimise with AdamW and a cosine learning rate schedule.
      6. Save the weights whenever validation AUC improves.
      7. Stop early if AUC stops improving for `patience` consecutive epochs.

    Args:
        real_manifest_paths: face_manifest.json paths for REAL (authentic) videos.
        fake_manifest_paths: face_manifest.json paths for FAKE (deepfake) videos.
        epochs:              Maximum number of training epochs.
        batch_size:          Samples per gradient update. 16 works on most CPUs;
                             use 32–64 if you have a GPU available.
        learning_rate:       Initial AdamW learning rate. 1e-4 is a safe default
                             for fine-tuning — large enough to adapt but small
                             enough not to overwrite useful pretrained features.
        val_split:           Fraction of total samples held out for validation.
        patience:            Stop training if val AUC does not improve for this
                             many consecutive epochs. Prevents overfitting.
        num_workers:         DataLoader worker processes. Keep at 0 on Windows
                             and macOS to avoid multiprocessing issues.

    Returns:
        Path to the saved fine-tuned weights (.pt file).
    """
    log    = logger.bind(component="trainer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training_device", device=str(device))

    # ── 1. Build datasets ──────────────────────────────────────────────────
    # We instantiate the dataset TWICE from the same manifests — once with
    # augmentations for training, once without for validation. Both objects
    # produce samples in the same deterministic order (no internal shuffle),
    # so index-based splitting gives consistent real/fake assignment to each set.

    train_dataset = FaceForensicsDataset(
        real_manifest_paths=real_manifest_paths,
        fake_manifest_paths=fake_manifest_paths,
        is_training=True,
    )
    val_dataset = FaceForensicsDataset(
        real_manifest_paths=real_manifest_paths,
        fake_manifest_paths=fake_manifest_paths,
        is_training=False,   # no augmentations on validation data
    )

    total = len(train_dataset)
    if total < 20:
        raise ValueError(
            f"Dataset too small ({total} samples). "
            "Run the Day 1 + Day 2 pipeline on more videos first. "
            "Aim for at least 50 face crops per class."
        )

    balance = train_dataset.class_balance()
    log.info("dataset_loaded", **balance)

    # ── 2. Split indices ───────────────────────────────────────────────────
    # torch.randperm with a fixed manual seed gives a reproducible shuffle.
    # Running fine_tune() twice with the same data will produce the same
    # train/val split, which is essential for comparing experiments fairly.
    n_val   = max(1, int(total * val_split))
    n_train = total - n_val

    generator   = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(total, generator=generator).tolist()

    train_indices = all_indices[:n_train]
    val_indices   = all_indices[n_train:]

    # Apply the same index partition to both dataset objects.
    # train_subset reads from train_dataset → augmentations ON.
    # val_subset reads from val_dataset → augmentations OFF.
    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset,   val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,                         # re-shuffle each epoch
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,                        # order doesn't matter for eval
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    log.info("data_split", train=n_train, val=n_val)

    # ── 3. Build model ─────────────────────────────────────────────────────
    model        = build_pytorch_model(pretrained=True)
    frozen_count = freeze_early_layers(model, freeze_blocks=freeze_blocks)
    model        = model.to(device)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "model_ready",
        frozen_params=frozen_count,
        trainable_params=trainable_count,
        freeze_blocks=freeze_blocks,
    )

    # ── 4. Loss, optimiser, scheduler ─────────────────────────────────────
    pos_weight = compute_pos_weight(train_dataset).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # We explicitly filter to only pass trainable parameters to AdamW.
    # Passing frozen parameters would raise no error, but it wastes memory
    # maintaining momentum buffers for parameters that will never be updated.
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # CosineAnnealingLR smoothly decays the learning rate from `learning_rate`
    # down to `eta_min` following a cosine curve over T_max epochs. This is
    # gentler than abrupt step-decay schedules and tends to find flatter
    # minima that generalise better.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── 5. Training loop ───────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_auc     = 0.0
    best_val_acc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    epochs_ran       = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        epoch_start = time.time()

        train_loss, train_acc, train_auc = _run_epoch(
            model, train_loader, criterion,
            optimizer=optimizer, is_training=True, device=device,
        )
        val_loss, val_acc, val_auc = _run_epoch(
            model, val_loader, criterion,
            optimizer=None, is_training=False, device=device,
        )

        scheduler.step()
        elapsed = time.time() - epoch_start

        log.info(
            "epoch",
            epoch=f"{epoch}/{epochs}",
            train_loss=round(train_loss, 4),
            train_acc=f"{train_acc:.1%}",
            train_auc=round(train_auc, 4),
            val_loss=round(val_loss, 4),
            val_acc=f"{val_acc:.1%}",
            val_auc=round(val_auc, 4),
            elapsed_s=round(elapsed, 1),
        )

        # We save on AUC improvement rather than loss improvement because
        # AUC directly measures the model's discrimination ability, which
        # is what matters for deepfake detection in practice.
        if best_epoch == 0 or val_auc > best_val_auc:
            best_val_auc     = val_auc
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(output_path))
            log.info(
                "best_model_saved",
                val_auc=round(best_val_auc, 4),
                path=str(output_path),
            )
        else:
            patience_counter += 1
            log.info("no_improvement", patience=f"{patience_counter}/{patience}")

            if patience_counter >= patience:
                log.info("early_stopping_triggered", epoch=epoch)
                break

    result = TrainingResult(
        weights_path=str(output_path),
        best_val_auc=float(best_val_auc),
        best_val_acc=float(best_val_acc),
        best_epoch=int(best_epoch),
        epochs_ran=int(epochs_ran),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        weight_decay=float(weight_decay),
        freeze_blocks=int(freeze_blocks),
        patience=int(patience),
    )

    if save_metadata:
        output_path.with_suffix(".json").write_text(json.dumps(asdict(result), indent=2))

    log.info(
        "fine_tuning_complete",
        best_val_auc=round(best_val_auc, 4),
        best_epoch=best_epoch,
        weights=str(output_path),
    )
    return output_path


def hyperparameter_tune(
    real_manifest_paths: List[Path],
    fake_manifest_paths: List[Path],
    epochs: int = 5,
    learning_rates: Iterable[float] = (3e-5, 1e-4, 3e-4),
    batch_sizes: Iterable[int] = (8, 16),
    weight_decays: Iterable[float] = (1e-5, 1e-4),
    freeze_blocks_options: Iterable[int] = (1, 2, 3),
    val_split: float = 0.2,
    patience: int = 2,
    num_workers: int = 0,
    output_dir: Path | None = None,
) -> dict:
    """
    Run a small grid search over fine-tuning hyperparameters.

    The best trial is selected by validation AUC, with validation accuracy as a
    tie-breaker. Its weights are copied to FINETUNED_WEIGHTS_PATH so the normal
    ONNX export and inference commands automatically use the tuned model.
    """
    import shutil

    log = logger.bind(component="hyperparameter_tuner")
    if output_dir is None:
        output_dir = FINETUNED_WEIGHTS_PATH.parent / "hyperparameter_trials"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    search_space = list(itertools.product(
        list(learning_rates),
        list(batch_sizes),
        list(weight_decays),
        list(freeze_blocks_options),
    ))

    if not search_space:
        raise ValueError("Hyperparameter search space is empty.")

    for trial_idx, (lr, batch_size, weight_decay, freeze_blocks) in enumerate(search_space, start=1):
        trial_name = (
            f"trial_{trial_idx:03d}_lr{lr:g}_bs{batch_size}_"
            f"wd{weight_decay:g}_freeze{freeze_blocks}"
        )
        trial_path = output_dir / f"{trial_name}.pt"

        log.info(
            "trial_start",
            trial=f"{trial_idx}/{len(search_space)}",
            learning_rate=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            freeze_blocks=freeze_blocks,
        )

        fine_tune(
            real_manifest_paths=real_manifest_paths,
            fake_manifest_paths=fake_manifest_paths,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            freeze_blocks=freeze_blocks,
            val_split=val_split,
            patience=patience,
            num_workers=num_workers,
            output_path=trial_path,
            save_metadata=True,
        )

        metadata_path = trial_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["trial"] = trial_idx
        metadata["trial_name"] = trial_name
        trials.append(metadata)

        log.info(
            "trial_complete",
            trial=trial_idx,
            val_auc=round(metadata["best_val_auc"], 4),
            val_acc=round(metadata["best_val_acc"], 4),
            weights=str(trial_path),
        )

    best = max(
        trials,
        key=lambda row: (
            row["best_val_auc"],
            row["best_val_acc"],
            -row["epochs_ran"],
        ),
    )

    shutil.copyfile(best["weights_path"], FINETUNED_WEIGHTS_PATH)
    summary = {
        "best_trial": best,
        "trials": sorted(
            trials,
            key=lambda row: (row["best_val_auc"], row["best_val_acc"]),
            reverse=True,
        ),
        "selection_metric": "max_val_auc_then_val_accuracy",
        "finetuned_weights_path": str(FINETUNED_WEIGHTS_PATH),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info(
        "hyperparameter_tuning_complete",
        best_trial=best["trial"],
        best_val_auc=round(best["best_val_auc"], 4),
        best_val_acc=round(best["best_val_acc"], 4),
        weights=str(FINETUNED_WEIGHTS_PATH),
    )
    return summary
