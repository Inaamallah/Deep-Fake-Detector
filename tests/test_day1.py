# tests/test_day1.py
"""
Day 1 unit tests.
Run with: pytest tests/test_day1.py -v
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import cv2

from src.ingestion.downloader import (
    _make_video_id,
    _is_url,
    _is_direct_video_url,
    _get_platform,
    _validate_video,
    VideoValidationError,
)
from src.ingestion.frame_extractor import (
    _frame_diff,
    _resize_frame,
    _uniform_indices,
    FrameBatch,
)
from config import settings


# ── Downloader unit tests ────────────────────────────────────────────────────

class TestVideoId:
    def test_stable(self):
        assert _make_video_id("https://example.com/a") == _make_video_id("https://example.com/a")

    def test_different_sources(self):
        assert _make_video_id("a") != _make_video_id("b")

    def test_length(self):
        assert len(_make_video_id("x")) == 16


class TestIsUrl:
    def test_https(self):
        assert _is_url("https://youtube.com/watch?v=abc") is True

    def test_http(self):
        assert _is_url("http://example.com/video.mp4") is True

    def test_local_path(self):
        assert _is_url("/home/user/video.mp4") is False

    def test_relative_path(self):
        assert _is_url("video.mp4") is False


class TestIsDirectVideoUrl:
    def test_mp4(self):
        assert _is_direct_video_url("https://cdn.example.com/clip.mp4") is True

    def test_webm(self):
        assert _is_direct_video_url("https://cdn.example.com/clip.webm") is True

    def test_youtube(self):
        assert _is_direct_video_url("https://youtube.com/watch?v=abc") is False


class TestGetPlatform:
    def test_youtube(self):
        assert _get_platform("https://www.youtube.com/watch?v=abc") == "youtube"

    def test_youtu_be(self):
        assert _get_platform("https://youtu.be/abc") == "youtube"

    def test_vimeo(self):
        assert _get_platform("https://vimeo.com/123") == "vimeo"

    def test_direct(self):
        assert _get_platform("https://cdn.example.com/video.mp4") == "direct"


class TestValidateVideo:
    BASE = {"duration_seconds": 30.0, "width": 1280, "height": 720,
            "fps": 30.0, "size_bytes": 10_000_000}

    def test_valid(self):
        _validate_video(self.BASE, "test")  # no exception

    def test_too_long(self):
        with pytest.raises(VideoValidationError, match="exceeds limit"):
            _validate_video({**self.BASE, "duration_seconds": 9999}, "test")

    def test_too_large(self):
        with pytest.raises(VideoValidationError, match="exceeds limit"):
            _validate_video({**self.BASE, "size_bytes": 999_999_999_999}, "test")

    def test_zero_duration(self):
        with pytest.raises(VideoValidationError, match="zero or negative"):
            _validate_video({**self.BASE, "duration_seconds": 0}, "test")

    def test_zero_dimensions(self):
        with pytest.raises(VideoValidationError, match="dimensions"):
            _validate_video({**self.BASE, "width": 0}, "test")

    def test_low_fps(self):
        with pytest.raises(VideoValidationError, match="FPS too low"):
            _validate_video({**self.BASE, "fps": 0.5}, "test")


# ── Frame extractor unit tests ───────────────────────────────────────────────

class TestFrameDiff:
    def test_identical_frames(self):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        assert _frame_diff(f, f) == pytest.approx(0.0, abs=1e-3)

    def test_different_frames(self):
        a = np.zeros((480, 640, 3), dtype=np.uint8)
        b = np.full((480, 640, 3), 200, dtype=np.uint8)
        diff = _frame_diff(a, b)
        assert diff > 100  # expect large diff

    def test_symmetric(self):
        a = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        b = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert abs(_frame_diff(a, b) - _frame_diff(b, a)) < 1e-3


class TestResizeFrame:
    def test_output_size(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = _resize_frame(frame, (640, 640))
        assert result.shape == (640, 640, 3)

    def test_no_stretch(self):
        # Letterboxed: wide frame into square canvas.
        # The image content should be centered with black bars top/bottom.
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)  # all white
        result = _resize_frame(frame, (200, 200))
        # Top row should be black (padding)
        assert result[0, 100, 0] == 0

    def test_square_input(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        result = _resize_frame(frame, (640, 640))
        assert result.shape == (640, 640, 3)


class TestUniformIndices:
    def test_correct_count(self):
        indices = _uniform_indices(300, 30.0, 2.0, 120)
        assert len(indices) <= 120

    def test_max_frames_respected(self):
        indices = _uniform_indices(30000, 30.0, 30.0, 50)
        assert len(indices) <= 50

    def test_starts_at_zero(self):
        indices = _uniform_indices(300, 30.0, 2.0, 120)
        assert indices[0] == 0


# ── Integration smoke test (requires ffmpeg + a real video) ──────────────────
# Skipped in CI if no network / ffmpeg available.

@pytest.mark.integration
def test_local_video_pipeline(tmp_path):
    """
    Create a synthetic video with OpenCV, run the full Day 1 pipeline on it.
    This validates the full download_video → extract_frames path without network.
    """
    # Create a 5-second, 640x480, 30fps synthetic video
    video_path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30, (640, 480))

    for i in range(150):  # 5 seconds at 30fps
        frame = np.full((480, 640, 3), dtype=np.uint8, fill_value=(i % 255))
        writer.write(frame)

    writer.release()
    assert video_path.exists()

    # Run the downloader (local path)
    from src.ingestion.downloader import download_video
    meta = download_video(str(video_path))
    assert meta.video_id is not None
    assert meta.duration_seconds > 0

    # Run frame extraction
    from src.ingestion.frame_extractor import extract_frames
    batch = extract_frames(
        video_path=meta.local_path,
        video_id=meta.video_id,
        duration_seconds=meta.duration_seconds,
        fps=meta.fps,
        strategy="uniform",
    )
    assert batch.count >= settings.min_frames
    assert all(p.exists() for p in batch.frame_paths)