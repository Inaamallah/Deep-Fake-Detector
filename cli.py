# cli.py
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

from config import settings
from src.ingestion.downloader import download_video, VideoDownloadError, VideoValidationError
from src.ingestion.frame_extractor import extract_frames, FrameExtractionError
from src.utils.logger import logger

console = Console()


@click.group()
def cli():
    """Deepfake Detector — production pipeline."""
    pass


@cli.command("calibrate-threshold")
@click.option(
    "--real-manifests", "real_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a REAL validation video. Repeat for multiple files.",
)
@click.option(
    "--fake-manifests", "fake_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a FAKE validation video. Repeat for multiple files.",
)
@click.option("--batch-size", default=8, type=int, help="Inference batch size (default: 8)")
@click.option("--output-dir", default="results/calibration", type=click.Path(),
              help="Directory for calibration JSON, CSV, and score plot.")
def calibrate_threshold(real_manifest_paths, fake_manifest_paths, batch_size, output_dir):
    """
    Tune the video-level fake threshold on labelled validation videos.

    Use videos that look like the new clips you care about. This command scores
    each full video, sweeps thresholds from 0.05 to 0.95, and recommends the
    threshold with the best F1 score.
    """
    from src.detection.evaluator import calibrate_video_threshold

    real_paths = [Path(p) for p in real_manifest_paths]
    fake_paths = [Path(p) for p in fake_manifest_paths]

    console.rule("[bold]Video Threshold Calibration[/bold]")
    console.print(f"  Real videos : {len(real_paths)}")
    console.print(f"  Fake videos : {len(fake_paths)}")
    console.print(f"  Batch size  : {batch_size}")

    try:
        result = calibrate_video_threshold(
            real_manifest_paths=real_paths,
            fake_manifest_paths=fake_paths,
            batch_size=batch_size,
            output_dir=Path(output_dir),
        )
    except Exception as e:
        console.print(f"[red]ERROR Calibration failed:[/red] {e}")
        raise SystemExit(1)

    best = result["best_metrics"]
    threshold = result["recommended_threshold"]

    console.print(f"\n[bold green]Recommended threshold: {threshold:.2f}[/bold green]")
    console.print(f"  F1                : {best['f1']:.4f}")
    console.print(f"  Balanced accuracy : {best['balanced_accuracy']:.4f}")
    console.print(f"  Precision         : {best['precision']:.4f}")
    console.print(f"  Recall            : {best['recall']:.4f}")
    console.print(f"  AUC               : {result['auc']:.4f}")
    console.print(f"  Artifacts         : {output_dir}")
    console.print("\nUse it like this:")
    console.print(f"  python cli.py infer <video_id> --threshold {threshold:.2f}")
    console.rule("[bold green]Calibration complete OK[/bold green]")


@cli.command("hyper-tune")
@click.option(
    "--real-manifests", "real_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a REAL training video. Repeat for multiple files.",
)
@click.option(
    "--fake-manifests", "fake_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a FAKE training video. Repeat for multiple files.",
)
@click.option("--epochs", default=5, type=int, help="Max epochs per trial (default: 5)")
@click.option("--patience", default=2, type=int, help="Early stopping patience per trial (default: 2)")
@click.option("--val-split", default=0.2, type=float, help="Validation fraction (default: 0.2)")
@click.option("--learning-rates", default="3e-5,1e-4,3e-4",
              help="Comma-separated learning rates.")
@click.option("--batch-sizes", default="8,16",
              help="Comma-separated batch sizes.")
@click.option("--weight-decays", default="1e-5,1e-4",
              help="Comma-separated AdamW weight decay values.")
@click.option("--freeze-blocks", default="1,2,3",
              help="Comma-separated EfficientNet block counts to freeze.")
@click.option("--output-dir", default="models/hyperparameter_trials", type=click.Path(),
              help="Directory for trial weights and summary.json.")
def hyper_tune(
    real_manifest_paths, fake_manifest_paths, epochs, patience, val_split,
    learning_rates, batch_sizes, weight_decays, freeze_blocks, output_dir,
):
    """
    Run a small hyperparameter search and install the best fine-tuned model.

    The best trial is selected by validation AUC, then validation accuracy. Its
    weights are copied to models/deepfake_efficientb4_finetuned.pt.
    """
    from src.detection.trainer import hyperparameter_tune

    def parse_grid(raw, cast):
        return [cast(item.strip()) for item in raw.split(",") if item.strip()]

    real_paths = [Path(p) for p in real_manifest_paths]
    fake_paths = [Path(p) for p in fake_manifest_paths]
    lr_grid = parse_grid(learning_rates, float)
    bs_grid = parse_grid(batch_sizes, int)
    wd_grid = parse_grid(weight_decays, float)
    freeze_grid = parse_grid(freeze_blocks, int)
    total_trials = len(lr_grid) * len(bs_grid) * len(wd_grid) * len(freeze_grid)

    console.rule("[bold]Hyperparameter Tuning[/bold]")
    console.print(f"  Real manifests : {len(real_paths)}")
    console.print(f"  Fake manifests : {len(fake_paths)}")
    console.print(f"  Trials         : {total_trials}")
    console.print(f"  Epochs/trial   : {epochs}")
    console.print(f"  Learning rates : {lr_grid}")
    console.print(f"  Batch sizes    : {bs_grid}")
    console.print(f"  Weight decays  : {wd_grid}")
    console.print(f"  Freeze blocks  : {freeze_grid}")

    try:
        summary = hyperparameter_tune(
            real_manifest_paths=real_paths,
            fake_manifest_paths=fake_paths,
            epochs=epochs,
            learning_rates=lr_grid,
            batch_sizes=bs_grid,
            weight_decays=wd_grid,
            freeze_blocks_options=freeze_grid,
            val_split=val_split,
            patience=patience,
            output_dir=Path(output_dir),
        )
    except Exception as e:
        console.print(f"[red]ERROR Hyperparameter tuning failed:[/red] {e}")
        raise SystemExit(1)

    best = summary["best_trial"]
    console.print("\n[bold green]Best trial installed[/bold green]")
    console.print(f"  Trial          : {best['trial_name']}")
    console.print(f"  Val AUC        : {best['best_val_auc']:.4f}")
    console.print(f"  Val accuracy   : {best['best_val_acc']:.4f}")
    console.print(f"  Best epoch     : {best['best_epoch']}")
    console.print(f"  Weights        : {summary['finetuned_weights_path']}")
    console.print(f"  Summary        : {Path(output_dir) / 'summary.json'}")
    console.print("\nRe-export ONNX with:")
    console.print("  python cli.py infer <video_id> --export-onnx")
    console.rule("[bold green]Hyperparameter tuning complete OK[/bold green]")


@cli.command()
@click.argument("source", metavar="VIDEO_URL_OR_PATH")
@click.option("--strategy", default="hybrid",
              type=click.Choice(["scene_change", "uniform", "hybrid"]),
              help="Frame sampling strategy.")
@click.option("--max-frames", default=None, type=int,
              help=f"Override max frames (default: {settings.max_frames})")
@click.option("--dry-run", is_flag=True,
              help="Download and probe video but skip frame extraction.")
def ingest(source: str, strategy: str, max_frames, dry_run: bool):
    """
    Download or load a video and extract frames.

    VIDEO_URL_OR_PATH can be:
      - A YouTube URL: https://youtube.com/watch?v=...
      - Any yt-dlp compatible URL (Vimeo, TikTok, Twitter, etc.)
      - A direct .mp4 / .webm URL
      - A local file path: /path/to/video.mp4

    Examples:

      python cli.py ingest https://www.youtube.com/watch?v=EXAMPLE

      python cli.py ingest /videos/suspect_clip.mp4 --strategy scene_change

      python cli.py ingest https://example.com/clip.mp4 --max-frames 60
    """

    if max_frames:
        settings.max_frames = max_frames

    console.rule("[bold]Step 1: Video Download[/bold]")

    # ── Download ─────────────────────────────────────────────────────────────
    try:
        with console.status("[cyan]Downloading / validating video...[/cyan]"):
            meta = download_video(source)
    except (VideoDownloadError, VideoValidationError) as e:
        console.print(f"[red]ERROR Download failed:[/red] {e}")
        raise SystemExit(1)

    # Print metadata table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    rows = [
        ("Video ID", meta.video_id),
        ("Duration", f"{meta.duration_seconds:.1f}s"),
        ("Resolution", f"{meta.width}x{meta.height}"),
        ("FPS", f"{meta.fps:.2f}"),
        ("Size", f"{meta.size_bytes / 1e6:.1f} MB"),
        ("Platform", meta.platform or "local"),
        ("Local path", str(meta.local_path)),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)

    if dry_run:
        console.print("[yellow]--dry-run set, skipping frame extraction.[/yellow]")
        return

    # ── Extract frames ────────────────────────────────────────────────────────
    console.rule("[bold]Step 2: Frame Extraction[/bold]")

    try:
        with console.status(f"[cyan]Extracting frames (strategy={strategy})...[/cyan]"):
            batch = extract_frames(
                video_path=meta.local_path,
                video_id=meta.video_id,
                duration_seconds=meta.duration_seconds,
                fps=meta.fps,
                strategy=strategy,
            )
    except FrameExtractionError as e:
        console.print(f"[red]ERROR Frame extraction failed:[/red] {e}")
        raise SystemExit(1)

    console.print(f"[green]OK Extracted {batch.count} frames[/green]")
    console.print(f"  Strategy : {batch.extraction_strategy}")
    console.print(f"  Output   : {batch.frames_dir}")
    console.print(f"  Time     : {batch.elapsed_seconds:.1f}s")

    # Save a manifest JSON for Day 2 to pick up
    manifest_path = batch.frames_dir / "manifest.json"
    manifest = {
        "video_id": meta.video_id,
        "source": meta.source,
        "frames_dir": str(batch.frames_dir),
        "frame_paths": [str(p) for p in batch.frame_paths],
        "frame_count": batch.count,
        "strategy": batch.extraction_strategy,
        "video": meta.to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    console.print(f"  Manifest : {manifest_path}")

    console.rule("[bold green]Day 1 pipeline complete OK[/bold green]")

    
# Add to cli.py — below the existing `ingest` command

@cli.command()
@click.argument("video_id")
@click.option("--output-size", default=112, type=int,
              help="Aligned crop size in pixels (default: 112)")
@click.option("--max-faces", default=3, type=int,
              help="Max faces to extract per frame (default: 3)")
def detect_faces(video_id: str, output_size: int, max_faces: int):
    """
    Run face detection + alignment on frames extracted for VIDEO_ID.

    VIDEO_ID is the hash printed after running `ingest`.

    Example:
      python cli.py detect-faces a3b9f2c1d4e5
    """
    from src.detection.face_extractor import extract_faces_from_manifest

    manifest_path = settings.raw_frames_dir / video_id / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]No manifest found for {video_id}[/red]")
        console.print(f"Run [bold]python cli.py ingest <source>[/bold] first.")
        raise SystemExit(1)

    console.rule("[bold]Day 2: Face Detection & Alignment[/bold]")

    try:
        with console.status("[cyan]Detecting and aligning faces...[/cyan]"):
            batch = extract_faces_from_manifest(
                manifest_path=manifest_path,
                output_size=output_size,
                max_faces_per_frame=max_faces,
            )
    except Exception as e:
        console.print(f"[red]ERROR Face extraction failed:[/red] {e}")
        raise SystemExit(1)

    console.print(f"[green]OK Saved {batch.count} face crops[/green]")
    console.print(f"  Rejected      : {batch.rejected}")
    console.print(f"  No-face frames: {batch.frames_with_no_face}")
    console.print(f"  Output dir    : {batch.crops_dir}")
    console.print(f"  Time          : {batch.elapsed_seconds:.1f}s")

    stats = batch.detector_stats
    console.print(
        f"  MediaPipe hits: {stats.get('mediapipe_hits', 0)} | "
        f"MTCNN hits: {stats.get('mtcnn_hits', 0)} | "
        f"Detection rate: {stats.get('detection_rate', 0):.1%}"
    )

    face_manifest = batch.crops_dir / "face_manifest.json"
    console.print(f"  Manifest      : {face_manifest}")
    console.rule("[bold green]Day 2 pipeline complete OK[/bold green]")

@cli.command()
@click.argument("video_id")
def status(video_id: str):
    """Check what frames have been extracted for a VIDEO_ID."""
    frames_dir = settings.raw_frames_dir / video_id
    if not frames_dir.exists():
        console.print(f"[red]No frames found for video_id: {video_id}[/red]")
        raise SystemExit(1)

    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        console.print_json(json.dumps(manifest, indent=2))
    else:
        frames = sorted(
            list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        )
        console.print(f"Found {len(frames)} frames in {frames_dir}")

# Add to cli.py

@cli.command()
@click.argument("video_id")
@click.option("--batch-size", default=8, type=int,
              help="Inference batch size (default: 8). Increase for faster CPUs.")
@click.option("--export-onnx", is_flag=True,
              help="Force re-export of PyTorch weights to ONNX even if .onnx exists.")
@click.option("--threshold", default=0.5, type=float,
              help="Video-level fake threshold (default: 0.5). Use calibrate-threshold to tune it.")
def infer(video_id: str, batch_size: int, export_onnx: bool, threshold: float):
    """
    Run deepfake inference on face crops for VIDEO_ID.

    Reads the face_manifest.json written by detect-faces and scores
    every crop. Writes inference_result.json to the same directory.

    Example:
      python cli.py infer a3b9f2c1d4e5
    """
    from src.detection.model_downloader import export_to_onnx
    from src.detection.inference_engine import DeepfakeInferenceEngine
    from src.detection.video_scorer import score_video_from_manifest

    face_manifest_path = settings.face_crops_dir / video_id / "face_manifest.json"
    if not face_manifest_path.exists():
        console.print(f"[red]No face manifest found for {video_id}[/red]")
        console.print("Run [bold]python cli.py detect-faces <video_id>[/bold] first.")
        raise SystemExit(1)

    console.rule("[bold]Day 3: Model Inference[/bold]")

    # Export ONNX if requested or not yet done
    if export_onnx:
        with console.status("[cyan]Exporting model to ONNX...[/cyan]"):
            export_to_onnx(force=True)

    try:
        with console.status("[cyan]Loading ONNX engine...[/cyan]"):
            engine = DeepfakeInferenceEngine(batch_size=batch_size)

        with console.status("[cyan]Running inference over face crops...[/cyan]"):
            result = score_video_from_manifest(
                face_manifest_path,
                engine=engine,
                threshold=threshold,
            )

    except Exception as e:
        console.print(f"[red]ERROR Inference failed:[/red] {e}")
        raise

    # Print verdict with colour
    colour_map = {"DEEPFAKE": "red", "REAL": "green", "INCONCLUSIVE": "yellow"}
    colour = colour_map.get(result.verdict, "yellow")
    console.print(f"\n[bold {colour}]  VERDICT: {result.verdict}[/bold {colour}]")
    if result.verdict == "INCONCLUSIVE":
        console.print(
            "  [yellow]WARNING: Confidence too low to call REAL or FAKE. "
            "Consider adding more frames or calibrating the threshold.[/yellow]"
        )
    console.print(f"  Weighted P(fake) : {result.weighted_prob_fake:.1%}")
    console.print(f"  Mean P(fake)     : {result.mean_prob_fake:.1%}")
    console.print(f"  Threshold        : {result.decision_threshold:.1%}")
    console.print(f"  Confidence       : {result.overall_confidence:.1%}")
    console.print(f"  Faces scored     : {result.total_faces_scored}")
    console.print(f"  Elapsed          : {result.elapsed_seconds:.1f}s")
    console.print(f"  Avg latency/face : "
                  f"{sum(s.latency_ms for s in result.face_scores) / max(1, len(result.face_scores)):.1f}ms")

    result_path = settings.face_crops_dir / video_id / "inference_result.json"
    console.print(f"  Result saved to  : {result_path}")
    console.rule("[bold green]Day 3 pipeline complete OK[/bold green]")
# ADD this at the bottom of cli.py, after the existing `infer` command.

@cli.command()
@click.option(
    "--real-manifests", "real_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a REAL video. Repeat for multiple files.",
)
@click.option(
    "--fake-manifests", "fake_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a FAKE video. Repeat for multiple files.",
)
@click.option("--epochs",      default=10,   type=int,   help="Max training epochs (default: 10)")
@click.option("--batch-size",  default=16,   type=int,   help="Batch size (default: 16)")
@click.option("--lr",          default=1e-4, type=float, help="Learning rate (default: 1e-4)")
@click.option("--val-split",   default=0.2,  type=float, help="Validation fraction (default: 0.2)")
@click.option("--patience",    default=3,    type=int,   help="Early stopping patience (default: 3)")
def finetune(
    real_manifest_paths, fake_manifest_paths,
    epochs, batch_size, lr, val_split, patience,
):
    """
    Fine-tune EfficientNet-B4 on your labelled deepfake dataset.

    Prerequisites: run `ingest` and `detect-faces` on both real and fake
    videos first, then provide the resulting face_manifest.json files here.

    After fine-tuning, re-export the ONNX model by running:
      python cli.py infer <video_id> --export-onnx

    \b
    Example with two real videos and two fake videos:
      python cli.py finetune \\
        --real-manifests data/face_crops/abc1/face_manifest.json \\
        --real-manifests data/face_crops/abc2/face_manifest.json \\
        --fake-manifests data/face_crops/def1/face_manifest.json \\
        --fake-manifests data/face_crops/def2/face_manifest.json
    """
    from src.detection.trainer import fine_tune

    real_paths = [Path(p) for p in real_manifest_paths]
    fake_paths = [Path(p) for p in fake_manifest_paths]

    console.rule("[bold]Day 3: Fine-tuning EfficientNet-B4[/bold]")
    console.print(f"  Real manifests : {len(real_paths)}")
    console.print(f"  Fake manifests : {len(fake_paths)}")
    console.print(f"  Epochs         : {epochs}")
    console.print(f"  Batch size     : {batch_size}")
    console.print(f"  Learning rate  : {lr}")
    console.print(f"  Val split      : {val_split:.0%}")
    console.print(f"  Patience       : {patience}")

    try:
        saved_path = fine_tune(
            real_manifest_paths=real_paths,
            fake_manifest_paths=fake_paths,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            val_split=val_split,
            patience=patience,
        )
    except Exception as e:
        console.print(f"[red]✗ Fine-tuning failed:[/red] {e}")
        raise SystemExit(1)

    console.print(f"\n[green]✓ Fine-tuning complete[/green]")
    console.print(f"  Weights saved : {saved_path}")
    console.print(
        "\n[dim]Re-export ONNX with your new weights by running:[/dim]"
    )
    console.print("  python cli.py infer <video_id> --export-onnx")
    console.rule("[bold green]Fine-tuning complete ✓[/bold green]")


@cli.command()
@click.argument("video_id")
@click.option("--top-heatmaps",  default=5,    type=int,
              help="Number of Grad-CAM heatmaps to generate (default: 5)")
@click.option("--smooth-window", default=5,    type=int,
              help="Temporal smoothing window in frames (default: 5)")
@click.option("--bootstrap",     default=2000, type=int,
              help="Bootstrap iterations for confidence intervals (default: 2000)")
def analyze(video_id: str, top_heatmaps: int, smooth_window: int, bootstrap: int):
    """
    Run Day 4 analysis on a scored video: temporal analysis, Grad-CAM
    heatmaps, and bootstrap confidence intervals.

    Reads:   data/face_crops/<VIDEO_ID>/inference_result.json
    Writes:  data/results/<VIDEO_ID>/final_report.json
             data/results/<VIDEO_ID>/heatmaps/*.png

    Prerequisites: run `ingest`, `detect-faces`, and `infer` first.

    \b
    Example:
      python cli.py analyze a3b9f2c1
    """
    from src.scoring.report_builder import build_report

    inference_path = settings.face_crops_dir / video_id / "inference_result.json"

    if not inference_path.exists():
        console.print(f"[red]No inference result found for {video_id}[/red]")
        console.print("Run [bold]python cli.py infer <video_id>[/bold] first.")
        raise SystemExit(1)

    console.rule("[bold]Day 4: Temporal Analysis & Explainability[/bold]")

    try:
        with console.status("[cyan]Building analysis report...[/cyan]"):
            report = build_report(
                inference_result_path = inference_path,
                top_n_heatmaps       = top_heatmaps,
                smooth_window        = smooth_window,
                n_bootstrap          = bootstrap,
            )
    except Exception as exc:
        console.print(f"[red]✗ Analysis failed:[/red] {exc}")
        raise SystemExit(1)

    colour_map = {"DEEPFAKE": "red", "REAL": "green", "INCONCLUSIVE": "yellow"}
    colour = colour_map.get(report.verdict, "yellow")
    t      = report.temporal
    ci     = report.confidence_interval

    console.print(f"\n[bold {colour}]  VERDICT : {report.verdict}[/bold {colour}]")
    console.print(f"  Weighted P(fake) : {report.weighted_prob_fake:.1%}")
    console.print(
        f"  95% CI           : "
        f"[{ci['ci_lower_95']:.3f}, {ci['ci_upper_95']:.3f}]"
    )
    console.print(f"  Bootstrap std    : {ci['bootstrap_std']:.4f}")
    console.print(f"  Temporal verdict : {t['temporal_verdict']}")
    console.print(
        f"  Suspicious frames: {t['suspicious_frame_ratio']:.1%} of total"
    )
    console.print(
        f"  Suspicious windows: {len(t['suspicious_windows'])} detected"
    )
    console.print(
        f"  Run-length score : {t['run_length_score']:.2f} "
        f"({'clustered' if t['run_length_score'] > 1.2 else 'scattered'})"
    )
    console.print(
        f"  Peak frame       : #{t['peak_frame_idx']} "
        f"(P(fake)={t['peak_score']:.3f})"
    )
    console.print(f"  Heatmaps saved   : {len(report.heatmaps)}")
    console.print(f"  Elapsed          : {report.elapsed_seconds:.1f}s")

    result_path = settings.results_dir / video_id / "final_report.json"
    console.print(f"\n  Report saved to  : {result_path}")
    console.print(f"\n  {ci['interpretation']}")

    console.rule("[bold green]Day 4 complete ✓[/bold green]")
    
@cli.command()
@click.option(
    "--real-manifests", "real_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a REAL video. Repeat for multiple files.",
)
@click.option(
    "--fake-manifests", "fake_manifest_paths",
    multiple=True, required=True, type=click.Path(exists=True),
    help="Path to a face_manifest.json for a FAKE video. Repeat for multiple files.",
)
@click.option("--batch-size",  default=16,   type=int,   help="Batch size (default: 16)")
@click.option("--val-split",   default=0.2,  type=float, help="Validation split (default: 0.2)")
@click.option("--test-split",  default=0.1,  type=float, help="Test split (default: 0.1)")
@click.option("--output-dir",  default=None, type=click.Path(),
              help="Directory to save confusion matrix plots (optional)")
def evaluate(
    real_manifest_paths, fake_manifest_paths,
    batch_size, val_split, test_split, output_dir,
):
    """
    Evaluate fine-tuned model on train/val/test splits.

    Computes accuracy, precision, recall, F1, AUC, and generates
    confusion matrices for each split.

    \b
    Example:
      python cli.py evaluate \\
        --real-manifests data/face_crops/abc1/face_manifest.json \\
        --fake-manifests data/face_crops/def1/face_manifest.json \\
        --output-dir results/
    """
    from src.detection.evaluator import evaluate_model

    real_paths = [Path(p) for p in real_manifest_paths]
    fake_paths = [Path(p) for p in fake_manifest_paths]

    console.rule("[bold]Evaluating Fine-tuned Model[/bold]")
    console.print(f"  Real manifests : {len(real_paths)}")
    console.print(f"  Fake manifests : {len(fake_paths)}")
    console.print(f"  Batch size     : {batch_size}")
    console.print(f"  Val split      : {val_split:.0%}")
    console.print(f"  Test split     : {test_split:.0%}")

    try:
        results = evaluate_model(
            real_manifest_paths=real_paths,
            fake_manifest_paths=fake_paths,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            output_dir=output_dir,
        )
    except Exception as e:
        console.print(f"[red]✗ Evaluation failed:[/red] {e}")
        raise SystemExit(1)

    # Print results table
    console.print("\n[bold cyan]Metrics by Split[/bold cyan]")
    
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Train", style="magenta")
    table.add_column("Val", style="green")
    table.add_column("Test", style="yellow")

    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc"]
    for metric in metrics_to_show:
        row = [metric.upper()]
        for split in ["train", "val", "test"]:
            value = results[split].get(metric, 0)
            row.append(f"{value:.4f}")
        table.add_row(*row)

    console.print(table)

    # Print confusion matrices as text
    console.print("\n[bold cyan]Confusion Matrices[/bold cyan]")
    for split in ["train", "val", "test"]:
        cm = results["confusion_matrices"][split]
        console.print(f"\n[bold]{split.upper()}[/bold]")
        console.print(f"  True Negatives  : {results[split]['true_negatives']}")
        console.print(f"  False Positives : {results[split]['false_positives']}")
        console.print(f"  False Negatives : {results[split]['false_negatives']}")
        console.print(f"  True Positives  : {results[split]['true_positives']}")

    if output_dir:
        console.print(f"\n[green]✓ Confusion matrix plots saved to: {output_dir}[/green]")

    console.rule("[bold green]Evaluation complete ✓[/bold green]")


if __name__ == "__main__":
    cli()
