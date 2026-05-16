# src/ingestion/downloader.py
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from tqdm import tqdm

from config import settings
from src.utils.logger import logger


class VideoDownloadError(Exception):
    """Raised when a video cannot be downloaded or fails validation."""


class VideoValidationError(VideoDownloadError):
    """Raised when a video passes download but fails content validation."""


@dataclass
class VideoMetadata:
    """Metadata about a successfully downloaded or validated local video."""
    source: str                         # original URL or file path string
    local_path: Path                    # path to the video file on disk
    duration_seconds: float
    width: int
    height: int
    fps: float
    size_bytes: int
    video_id: str                       # sha256 of source string (stable ID)
    title: Optional[str] = None
    uploader: Optional[str] = None
    platform: Optional[str] = None
    download_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["local_path"] = str(self.local_path)
        return d

    def __str__(self) -> str:
        return (
            f"VideoMetadata(id={self.video_id[:8]}… "
            f"dur={self.duration_seconds:.1f}s "
            f"{self.width}x{self.height} @ {self.fps:.2f}fps "
            f"size={self.size_bytes / 1e6:.1f}MB)"
        )


def _make_video_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _is_url(source: str) -> bool:
    try:
        result = urlparse(source)
        return result.scheme in ("http", "https")
    except ValueError:
        return False


def _probe_video(path: Path) -> dict:
    """
    Use ffprobe to extract video stream metadata.
    Returns dict with duration, width, height, fps, size.
    Raises VideoValidationError if ffprobe fails or no video stream found.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise VideoValidationError(
            "ffprobe not found. Install ffmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        raise VideoValidationError(f"ffprobe timed out on {path}")

    if result.returncode != 0:
        raise VideoValidationError(
            f"ffprobe failed on {path}: {result.stderr.strip()}"
        )

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VideoValidationError(f"ffprobe returned invalid JSON: {e}")

    # Find first video stream
    video_streams = [
        s for s in probe.get("streams", [])
        if s.get("codec_type") == "video"
    ]
    if not video_streams:
        raise VideoValidationError(f"No video stream found in {path}")

    stream = video_streams[0]

    # Parse FPS (can be "30000/1001" rational or "30" string)
    raw_fps = stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, raw_fps.split("/"))
        fps = num / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        fps = float(raw_fps) if raw_fps else 0.0

    duration = float(probe.get("format", {}).get("duration", 0))
    size = int(probe.get("format", {}).get("size", path.stat().st_size))

    return {
        "duration_seconds": duration,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "size_bytes": size,
    }


def _validate_video(meta: dict, source: str) -> None:
    """
    Enforce business rules on video before processing.
    Raises VideoValidationError with a human-readable message.
    """
    dur = meta["duration_seconds"]
    size_mb = meta["size_bytes"] / 1e6

    if dur <= 0:
        raise VideoValidationError(f"Video has zero or negative duration: {source}")

    if dur > settings.max_video_duration_seconds:
        raise VideoValidationError(
            f"Video is {dur:.0f}s — exceeds limit of "
            f"{settings.max_video_duration_seconds}s. "
            f"Trim or use a shorter clip."
        )

    if size_mb > settings.max_video_size_mb:
        raise VideoValidationError(
            f"Video is {size_mb:.0f}MB — exceeds limit of "
            f"{settings.max_video_size_mb}MB."
        )

    if meta["width"] == 0 or meta["height"] == 0:
        raise VideoValidationError(
            f"Could not determine video dimensions for {source}"
        )

    if meta["fps"] < 1:
        raise VideoValidationError(
            f"Video FPS too low ({meta['fps']:.2f}) — possibly corrupt."
        )


def _download_direct_url(url: str, dest: Path) -> None:
    """
    Download a direct video URL (not YouTube) via httpx with progress bar.
    Used for .mp4, .mov, .webm direct links.
    """
    log = logger.bind(url=url, dest=str(dest))
    log.info("downloading_direct_url")

    try:    
        with httpx.stream("GET", url, follow_redirects=True,
                        timeout=settings.download_timeout_seconds) as r:

            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc="Downloading", leave=False
            ) as pbar:
            
                for chunk in r.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))

    except httpx.HTTPStatusError as e:
        raise VideoDownloadError(f"HTTP error: {e}")

    except httpx.RequestError as e:
        raise VideoDownloadError(f"Network error: {e}")


def _download_with_ytdlp(url: str, dest_dir: Path) -> Path:
    """
    Use yt-dlp to download from YouTube, Vimeo, TikTok, Twitter, etc.
    Returns path to the downloaded file.

    yt-dlp format string explanation:
      bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best
      → 720p max (enough for face detection, keeps files small on CPU machines)
      → prefer mp4 container for maximum ffmpeg compatibility
    """
    log = logger.bind(url=url)
    log.info("downloading_via_ytdlp")

    output_template = str(dest_dir / "%(id)s.%(ext)s")

    # Resolve yt-dlp executable: prefer the venv's Scripts/yt-dlp.exe,
    # fall back to sys.executable -m yt_dlp (always works if yt-dlp is
    # installed in the same Python environment the worker is running).
    ytdlp_path = shutil.which("yt-dlp")
    if ytdlp_path:
        ytdlp_cmd = [ytdlp_path]
    else:
        ytdlp_cmd = [sys.executable, "-m", "yt_dlp"]

    cmd = [
        *ytdlp_cmd,
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",           # never download a whole playlist accidentally
        "--max-filesize", f"{settings.max_video_size_mb}m",
        "--socket-timeout", "30",
        "--retries", "3",
        "--print", "after_move:filepath",   # prints the final path to stdout
        url,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=settings.download_timeout_seconds + 60
        )
    except FileNotFoundError:
        raise VideoDownloadError(
            "yt-dlp not found. Install with: pip install yt-dlp "
            "(make sure you are using the same Python environment as the worker)"
        )
    except subprocess.TimeoutExpired:
        raise VideoDownloadError(f"yt-dlp timed out downloading {url}")

    if result.returncode != 0:
        raise VideoDownloadError(
            f"yt-dlp failed: {result.stderr.strip()}"
        )

    # The last non-empty line of stdout is the file path
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        raise VideoDownloadError(
            "yt-dlp did not print a file path. Output: " + result.stdout[:500]
        )

    downloaded_path = Path(lines[-1])
    if not downloaded_path.exists():
        raise VideoDownloadError(
            f"yt-dlp reported path {downloaded_path} but file not found."
        )

    log.info("ytdlp_download_complete", path=str(downloaded_path))
    return downloaded_path


def _is_direct_video_url(url: str) -> bool:
    """Heuristic: does the URL path end with a known video extension?"""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"))


def _get_platform(url: str) -> str:
    host = urlparse(url).netloc.lower()
    mapping = {
        "youtube.com": "youtube", "youtu.be": "youtube",
        "vimeo.com": "vimeo",
        "tiktok.com": "tiktok",
        "twitter.com": "twitter", "x.com": "twitter",
        "instagram.com": "instagram",
        "facebook.com": "facebook", "fb.watch": "facebook",
        "twitch.tv": "twitch",
    }
    for domain, platform in mapping.items():
        if domain in host:
            return platform
    return "direct"


def download_video(source: str) -> VideoMetadata:
    """
    Main entry point. Accepts a URL or local file path.

    Returns VideoMetadata on success.
    Raises VideoDownloadError or VideoValidationError on failure.

    Examples:
        meta = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        meta = download_video("/path/to/local/video.mp4")
        meta = download_video("https://example.com/interview.mp4")
    """
    log = logger.bind(source=source)
    video_id = _make_video_id(source)
    log = log.bind(video_id=video_id)

    # ── LOCAL FILE PATH ──────────────────────────────────────────────────────
    if not _is_url(source):
        local_path = Path(source).expanduser().resolve()
        if not local_path.exists():
            raise VideoDownloadError(f"Local file not found: {local_path}")
        if not local_path.is_file():
            raise VideoDownloadError(f"Not a file: {local_path}")

        log.info("using_local_file", path=str(local_path))
        probe = _probe_video(local_path)
        _validate_video(probe, source)

        return VideoMetadata(
            source=source,
            local_path=local_path,
            video_id=video_id,
            **probe,
        )

    # ── URL ──────────────────────────────────────────────────────────────────
    platform = _get_platform(source)
    log = log.bind(platform=platform)

    # Temporary directory for download (cleaned up on failure)
    tmp_dir = settings.data_dir / "downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log.info("starting_download")
    start = time.time()

    try:
        if _is_direct_video_url(source):
            # Direct .mp4 / .webm URL
            ext = Path(urlparse(source).path).suffix or ".mp4"
            dest = tmp_dir / f"{video_id}{ext}"
            _download_direct_url(source, dest)
            local_path = dest
        else:
            # Platform URL — use yt-dlp
            local_path = _download_with_ytdlp(source, tmp_dir)

    except (httpx.HTTPError, httpx.RequestError) as e:
        raise VideoDownloadError(f"HTTP error downloading {source}: {e}") from e

    elapsed = time.time() - start
    log.info("download_complete", elapsed_s=round(elapsed, 2), path=str(local_path))

    probe = _probe_video(local_path)
    _validate_video(probe, source)

    log.info("video_validated", **probe)

    return VideoMetadata(
        source=source,
        local_path=local_path,
        video_id=video_id,
        platform=platform,
        **probe,
    )