# tests/test_day5.py  ← NEW FILE
"""
Day 5 unit and integration tests.
Run: pytest tests/test_day5.py -v

The database tests use a temporary SQLite file so they do not pollute
the real jobs.db. The route tests mock run_pipeline.delay so they do
not require a running Redis or Celery worker.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """
    Override the database path to use a temp file for each test.
    monkeypatch replaces the module-level DB_PATH without affecting other tests.
    """
    import src.api.database as db_module
    db_path = tmp_path / "test_jobs.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.init_db()
    return db_path


@pytest.fixture()
def api_client(temp_db):
    """
    TestClient with the FastAPI app.
    Mocks run_pipeline.delay so no Celery/Redis is needed.
    The X-API-Key header is set to the default dev key.
    """
    with patch("src.api.routes.run_pipeline") as mock_task:
        mock_task.delay = MagicMock(return_value=None)
        from src.api.app import create_app
        app    = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        client.headers.update({"X-API-Key": "dev-key-change-in-production"})
        yield client, mock_task


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------

class TestSubmitURLRequest:
    def test_valid_https_url(self):
        from src.api.models import SubmitURLRequest
        req = SubmitURLRequest(url="https://youtube.com/watch?v=abc")
        assert req.url == "https://youtube.com/watch?v=abc"

    def test_valid_http_url(self):
        from src.api.models import SubmitURLRequest
        req = SubmitURLRequest(url="http://example.com/video.mp4")
        assert req.url.startswith("http")

    def test_missing_scheme_raises(self):
        from pydantic import ValidationError
        from src.api.models import SubmitURLRequest
        with pytest.raises(ValidationError, match="http"):
            SubmitURLRequest(url="youtube.com/watch?v=abc")

    def test_strips_whitespace(self):
        from src.api.models import SubmitURLRequest
        req = SubmitURLRequest(url="  https://example.com/v.mp4  ")
        assert not req.url.startswith(" ")
        assert not req.url.endswith(" ")


class TestJobStatus:
    def test_str_values_are_uppercase(self):
        from src.api.models import JobStatus
        for status in JobStatus:
            assert status.value == status.value.upper()

    def test_done_and_failed_are_terminal(self):
        from src.api.models import JobStatus
        terminal = {JobStatus.DONE, JobStatus.FAILED}
        assert JobStatus.DONE in terminal
        assert JobStatus.FAILED in terminal
        assert JobStatus.PENDING not in terminal


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_create_job_returns_uuid(self, temp_db):
        from src.api.database import create_job
        import re
        job_id = create_job(source="https://example.com/v.mp4")
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(job_id)

    def test_get_job_returns_none_for_unknown(self, temp_db):
        from src.api.database import get_job
        assert get_job("nonexistent-id") is None

    def test_get_job_after_create(self, temp_db):
        from src.api.database import create_job, get_job
        from src.api.models import JobStatus
        job_id = create_job(source="https://example.com/v.mp4")
        job    = get_job(job_id)
        assert job is not None
        assert job["job_id"]  == job_id
        assert job["status"]  == JobStatus.PENDING.value
        assert job["source"]  == "https://example.com/v.mp4"
        assert job["video_id"] is None

    def test_update_job_status(self, temp_db):
        from src.api.database import create_job, get_job, update_job
        from src.api.models import JobStatus
        job_id = create_job(source="https://example.com/v.mp4")
        update_job(job_id, JobStatus.DOWNLOADING)
        job = get_job(job_id)
        assert job["status"] == JobStatus.DOWNLOADING.value

    def test_update_job_sets_video_id(self, temp_db):
        from src.api.database import create_job, get_job, update_job
        from src.api.models import JobStatus
        job_id = create_job(source="https://example.com/v.mp4")
        update_job(job_id, JobStatus.EXTRACTING_FRAMES, video_id="abc123")
        job = get_job(job_id)
        assert job["video_id"] == "abc123"

    def test_update_job_does_not_clear_existing_video_id(self, temp_db):
        """COALESCE logic: passing video_id=None should not overwrite an existing value."""
        from src.api.database import create_job, get_job, update_job
        from src.api.models import JobStatus
        job_id = create_job(source="https://example.com/v.mp4")
        update_job(job_id, JobStatus.EXTRACTING_FRAMES, video_id="abc123")
        update_job(job_id, JobStatus.DETECTING_FACES, video_id=None)
        job = get_job(job_id)
        assert job["video_id"] == "abc123"   # should still be there

    def test_update_job_error_message(self, temp_db):
        from src.api.database import create_job, get_job, update_job
        from src.api.models import JobStatus
        job_id = create_job(source="bad_source")
        update_job(job_id, JobStatus.FAILED, error_message="Download failed")
        job = get_job(job_id)
        assert job["error_message"] == "Download failed"

    def test_updated_at_increases(self, temp_db):
        from src.api.database import create_job, get_job, update_job
        from src.api.models import JobStatus
        job_id    = create_job(source="https://example.com/v.mp4")
        created   = get_job(job_id)["created_at"]
        time.sleep(0.01)   # ensure clock advances
        update_job(job_id, JobStatus.DOWNLOADING)
        updated   = get_job(job_id)["updated_at"]
        assert updated >= created


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, api_client):
        client, _ = api_client
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_no_auth_required(self, api_client):
        client, _ = api_client
        # Remove the API key and health should still work
        resp = client.get(
            "/api/v1/health", headers={"X-API-Key": ""}
        )
        # Health has no auth dependency so it should return 200
        assert resp.status_code == 200


class TestAuthMiddleware:
    def test_missing_key_returns_403(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://example.com/v.mp4"},
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://example.com/v.mp4"},
            headers={"X-API-Key": "completely-wrong-key"},
        )
        assert resp.status_code == 403

    def test_correct_key_passes(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://youtube.com/watch?v=test"},
        )
        # 202 = Accepted — job queued successfully
        assert resp.status_code == 202


class TestSubmitURL:
    def test_returns_202_with_job_id(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://youtube.com/watch?v=test"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "PENDING"

    def test_celery_task_is_called(self, api_client):
        client, mock_task = api_client
        client.post(
            "/api/v1/jobs/url",
            json={"url": "https://youtube.com/watch?v=test"},
        )
        mock_task.delay.assert_called_once()

    def test_invalid_url_returns_422(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "not-a-url"},
        )
        assert resp.status_code == 422

    def test_missing_url_returns_422(self, api_client):
        client, _ = api_client
        resp = client.post("/api/v1/jobs/url", json={})
        assert resp.status_code == 422


class TestGetJobStatus:
    def test_returns_404_for_unknown_job(self, api_client):
        client, _ = api_client
        resp = client.get("/api/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_returns_pending_for_new_job(self, api_client):
        client, _ = api_client
        submit = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://youtube.com/watch?v=test"},
        )
        job_id = submit.json()["job_id"]
        poll   = client.get(f"/api/v1/jobs/{job_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] == "PENDING"
        assert poll.json()["job_id"] == job_id


class TestGetResult:
    def test_returns_404_for_unknown_job(self, api_client):
        client, _ = api_client
        resp = client.get("/api/v1/jobs/nonexistent-id/result")
        assert resp.status_code == 404

    def test_pending_job_returns_null_report(self, api_client):
        client, _ = api_client
        submit = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://youtube.com/watch?v=test"},
        )
        job_id = submit.json()["job_id"]
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        body = result.json()
        assert body["status"]  == "PENDING"
        assert body["report"]  is None

    def test_failed_job_returns_error_message(self, api_client, temp_db):
        from src.api.database import create_job, update_job
        from src.api.models import JobStatus
        client, _ = api_client
        job_id = create_job(source="bad_url")
        update_job(job_id, JobStatus.FAILED, error_message="Download failed: 404")
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        body = result.json()
        assert body["status"]        == "FAILED"
        assert "Download failed" in body["error_message"]

    def test_done_job_returns_report(self, api_client, temp_db, tmp_path):
        from src.api.database import create_job, update_job
        from src.api.models import JobStatus
        client, _ = api_client

        # Write a fake final_report.json
        report_data = {"verdict": "REAL", "weighted_prob_fake": 0.12}
        report_path = tmp_path / "final_report.json"
        report_path.write_text(json.dumps(report_data))

        job_id = create_job(source="https://example.com/v.mp4")
        update_job(
            job_id,
            JobStatus.DONE,
            video_id="abc123",
            result_path=str(report_path),
        )

        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        body = result.json()
        assert body["status"]            == "DONE"
        assert body["report"]["verdict"] == "REAL"