# tests/test_smoke.py  ← NEW FILE
"""
Smoke tests that verify the deployed system end-to-end.

These tests are different from the unit tests in tests/test_day*.py.
Unit tests are fast, isolated, and use mocks. Smoke tests are slower,
require a running system, and test real integration between components.

Run only when the Docker stack is up:
    pytest tests/test_smoke.py -v -m smoke

To skip these in CI when Docker is not available:
    pytest tests/ -v -m "not smoke"
"""
import time
import os

import pytest
import httpx

# Skip all smoke tests unless explicitly opted in.
# This prevents them from running accidentally during unit test passes.
pytestmark = pytest.mark.smoke

BASE_URL = os.getenv("SMOKE_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "dev-key-change-in-production")
HEADERS  = {"X-API-Key": API_KEY}

# A short, publicly available video for testing.
# Rick Astley's "Never Gonna Give You Up" — stable, short, always available.
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture(scope="module")
def client():
    """HTTP client that attaches the API key to every request."""
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=30) as c:
        yield c


class TestHealthAndAuth:
    def test_health_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_missing_api_key_returns_403(self, client):
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": TEST_URL},
            headers={"X-API-Key": ""},  # override the fixture's key
        )
        assert resp.status_code == 403

    def test_wrong_api_key_returns_403(self, client):
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": TEST_URL},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_metrics_endpoint_returns_prometheus_text(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Prometheus text format always starts with a HELP comment
        assert b"# HELP" in resp.content


class TestJobSubmission:
    def test_submit_url_returns_202_and_job_id(self, client):
        resp = client.post("/api/v1/jobs/url", json={"url": TEST_URL})
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id"   in body
        assert body["status"] == "PENDING"
        assert len(body["job_id"]) > 10   # should be a UUID

    def test_invalid_url_format_returns_422(self, client):
        resp = client.post("/api/v1/jobs/url", json={"url": "not-a-url"})
        assert resp.status_code == 422

    def test_missing_url_field_returns_422(self, client):
        resp = client.post("/api/v1/jobs/url", json={})
        assert resp.status_code == 422


class TestJobPolling:
    def test_unknown_job_id_returns_404(self, client):
        resp = client.get("/api/v1/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_result_for_unknown_job_returns_404(self, client):
        resp = client.get("/api/v1/jobs/nonexistent-job-id/result")
        assert resp.status_code == 404

    def test_new_job_is_initially_pending_or_running(self, client):
        """
        Submit a job, then immediately poll it.
        It should be in PENDING or one of the running states — not DONE yet.
        """
        submit_resp = client.post("/api/v1/jobs/url", json={"url": TEST_URL})
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        # Poll immediately — job should be in an early state
        poll_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert poll_resp.status_code == 200
        status = poll_resp.json()["status"]
        assert status in ("PENDING", "DOWNLOADING", "EXTRACTING_FRAMES",
                          "DETECTING_FACES", "RUNNING_INFERENCE", "ANALYZING")

    def test_result_for_pending_job_has_null_report(self, client):
        submit_resp = client.post("/api/v1/jobs/url", json={"url": TEST_URL})
        job_id = submit_resp.json()["job_id"]

        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result_resp.status_code == 200
        body = result_resp.json()
        # Job is still running so the report field should be null
        assert body["report"] is None

    @pytest.mark.slow
    def test_full_pipeline_completes_and_returns_verdict(self, client):
        """
        End-to-end test: submit a job, poll until DONE (up to 10 minutes),
        then verify the report contains a valid verdict.

        Mark as `slow` so it can be excluded from quick smoke test runs:
            pytest -m "smoke and not slow"
        """
        submit_resp = client.post("/api/v1/jobs/url", json={"url": TEST_URL})
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        # Poll with a 600s (10 minute) total timeout and 5s intervals.
        deadline = time.time() + 600
        final_status = None

        while time.time() < deadline:
            time.sleep(5)
            poll = client.get(f"/api/v1/jobs/{job_id}")
            assert poll.status_code == 200
            final_status = poll.json()["status"]
            if final_status in ("DONE", "FAILED"):
                break

        assert final_status == "DONE", (
            f"Job did not complete within 10 minutes. Last status: {final_status}"
        )

        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        report = result.json()["report"]
        assert report is not None
        assert report["verdict"] in ("REAL", "DEEPFAKE", "INCONCLUSIVE")
        assert "weighted_prob_fake" in report
        assert "temporal"           in report
        assert "confidence_interval" in report