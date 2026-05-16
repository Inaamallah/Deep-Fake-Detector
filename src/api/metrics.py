# src/api/metrics.py  ← NEW FILE

"""
Prometheus metrics instrumentation for the deepfake detection API.

Why instrument your own service rather than relying on Nginx logs alone?
Nginx can tell you HTTP status codes and response times at the network level,
but it cannot tell you things like "how many jobs are currently in PENDING state"
or "what percentage of inference runs are returning DEEPFAKE verdicts".
Application-level metrics capture the business logic layer.

Usage: the /metrics endpoint is exposed by the FastAPI app (added in app.py).
Point your Prometheus server at http://your-api:8000/metrics to scrape it.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# We use a custom registry rather than the default global one.
# The default registry includes Python process metrics (GC, memory, threads)
# which are useful but can cause issues in multi-process Uvicorn deployments
# because each worker would register the same metric names. A custom registry
# avoids that collision and keeps our metrics clean and predictable.
REGISTRY = CollectorRegistry(auto_describe=True)

# ── Request metrics ──────────────────────────────────────────────────────────

# http_requests_total: a counter that increments for every HTTP request.
# Labels (method, endpoint, status_code) let you slice the data in queries.
# In Prometheus query language you would write:
#   rate(http_requests_total{status_code="200"}[5m])
# to get requests per second with a 200 status over the last 5 minutes.
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    labelnames=["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

# http_request_duration_seconds: a histogram that records the distribution
# of response times. Histograms are more useful than averages because they
# let you compute percentiles: p50 (median), p95, p99.
# The buckets define where the histogram "bins" are. We choose buckets that
# are meaningful for an API that proxies long-running ML work:
# most health checks finish in <0.05s; most analysis jobs take >30s.
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0],
    registry=REGISTRY,
)

# ── Job metrics ──────────────────────────────────────────────────────────────

jobs_submitted_total = Counter(
    "jobs_submitted_total",
    "Total jobs submitted (URL + upload combined)",
    registry=REGISTRY,
)

jobs_completed_total = Counter(
    "jobs_completed_total",
    "Jobs that reached DONE or FAILED status",
    labelnames=["final_status"],   # "DONE" or "FAILED"
    registry=REGISTRY,
)

# verdict_total counts real vs deepfake verdicts over time.
# This is a business metric — if the ratio shifts dramatically,
# it might indicate a change in the input distribution or a model issue.
verdict_total = Counter(
    "verdict_total",
    "Deepfake detection verdicts issued",
    labelnames=["verdict"],   # "REAL", "DEEPFAKE", "INCONCLUSIVE"
    registry=REGISTRY,
)

# jobs_in_progress: a gauge (can go up AND down) counting currently
# running pipeline tasks. A sustained high value means your workers
# are overwhelmed and jobs are queuing up.
jobs_in_progress = Gauge(
    "jobs_in_progress",
    "Number of pipeline tasks currently executing",
    registry=REGISTRY,
)

# ── Inference metrics ────────────────────────────────────────────────────────

inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Time taken for the full pipeline (download through report generation)",
    buckets=[10, 30, 60, 120, 180, 300, 600],
    registry=REGISTRY,
)


def metrics_response():
    """
    Generate the Prometheus text exposition format.
    Call this in the /metrics route handler.
    Returns (content: bytes, content_type: str).
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST