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
        frames = sorted(frames_dir.glob("*.png"))
        console.print(f"Found {len(frames)} frames in {frames_dir}")


if __name__ == "__main__":
    cli()
