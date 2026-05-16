# src/api/middleware.py  ← NEW FILE

"""
Structured request logging and metrics middleware.

FastAPI middleware wraps every single request that goes through the server.
Think of it as a function that runs before AND after your route handler,
giving you a hook to measure timing, log context, and record metrics
without touching any individual route.

Why middleware rather than adding logging to each route?
Because cross-cutting concerns (logging, timing, authentication headers)
should not be repeated in every route function. If you later want to add
a request ID to every log line, you change this one file rather than 50
route functions.
"""
import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

logger = structlog.get_logger()

# Endpoints that are polled constantly and would flood your logs if logged
# at INFO level. We still record their metrics but skip the log line.
_SILENT_PATHS = {"/api/v1/health", "/metrics"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    For every request:
      1. Assigns a unique request_id (UUID4) for distributed tracing.
      2. Binds the request_id to structlog's context vars so every log
         line emitted during that request automatically includes it.
      3. Times the request.
      4. Logs the completed request with method, path, status, and duration.
      5. Records Prometheus metrics.

    The request_id is also returned in the X-Request-ID response header
    so clients can include it in bug reports for easy log correlation.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]  # short 8-char prefix is enough for correlation

        # Bind the request_id to structlog's context variables.
        # Any structlog logger called anywhere during this request's
        # lifecycle will automatically include request_id in its output.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()

        # call_next runs the actual route handler (and any inner middleware).
        # We wrap it in try/except so that even if the handler raises an
        # unhandled exception, we still record metrics and log the failure.
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error(
                "request_unhandled_exception",
                duration_ms=round(duration * 1000, 2),
                error=str(exc),
            )
            # Re-raise so FastAPI's default exception handler takes over.
            raise

        duration = time.perf_counter() - start
        status   = response.status_code

        # Normalise the path for Prometheus labels.
        # Without normalisation, paths like /api/v1/jobs/uuid1 and
        # /api/v1/jobs/uuid2 would create separate metric label combinations,
        # leading to "label cardinality explosion" — millions of unique
        # label combinations that consume enormous memory in Prometheus.
        # We replace UUIDs with {id} to group them into one label value.
        normalised_path = _normalise_path(request.url.path)

        # Record Prometheus metrics (always, even for silent paths)
        http_requests_total.labels(
            method=request.method,
            endpoint=normalised_path,
            status_code=str(status),
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=normalised_path,
        ).observe(duration)

        # Log the completed request (skip noisy health/metrics endpoints)
        if request.url.path not in _SILENT_PATHS:
            log_fn = logger.warning if status >= 500 else logger.info
            log_fn(
                "request_completed",
                status=status,
                duration_ms=round(duration * 1000, 2),
            )

        # Attach the request ID to the response so clients can correlate
        response.headers["X-Request-ID"] = request_id
        return response


def _normalise_path(path: str) -> str:
    """
    Replace path segments that look like UUIDs or video IDs with {id}.

    Examples:
      /api/v1/jobs/550e8400-e29b-41d4-a716-446655440000  →  /api/v1/jobs/{id}
      /api/v1/jobs/a3b9f2c1/result                        →  /api/v1/jobs/{id}/result
    """
    import re
    # Match UUID4 format (xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx)
    path = re.sub(
        r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "/{id}",
        path,
    )
    # Match short hex IDs (8-16 hex chars) — our video_id format
    path = re.sub(r"/[0-9a-f]{8,16}(?=/|$)", "/{id}", path)
    return path