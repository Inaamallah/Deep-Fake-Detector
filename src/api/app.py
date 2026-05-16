# src/api/app.py  ← NEW FILE
"""
FastAPI application factory.

We use the factory pattern (create_app() returns the app) rather than
creating `app` directly at module level. This makes testing cleaner because
tests can call create_app() with different configurations without
module-level side effects running at import time.
"""
from fastapi.responses import Response
from src.api.middleware import RequestLoggingMiddleware
from src.api.metrics import metrics_response

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from src.api.database import init_db
from src.api.limiter import limiter
from src.api.routes import router
from src.utils.logger import logger


def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        The lifespan context manager replaces the old @app.on_event("startup")
        pattern from FastAPI < 0.93. Everything before `yield` runs on startup;
        everything after runs on shutdown.

        Startup: initialise the SQLite database (creates the jobs table if
        it does not already exist). This is idempotent and fast (<1ms).
        """
        logger.info("api_startup")
        init_db()
        logger.info("database_ready", db="jobs.db")
        yield
        logger.info("api_shutdown")

    app = FastAPI(
        title       = "Deepfake Detection API",
        description = (
            "Production-grade deepfake detection pipeline. "
            "Submit a video URL or upload a file, poll for results, "
            "and retrieve the full analysis report with Grad-CAM heatmaps."
        ),
        version  = "1.0.0",
        lifespan = lifespan,
        # OpenAPI docs are available at /docs (Swagger UI) and /redoc.
        # In a production deployment you would disable these or restrict
        # them to internal network access.
        docs_url  = "/docs",
        redoc_url = "/redoc",
    )

    # Rate limiting exception handler — must be registered before routes
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware — allows the React frontend (Day 6) running on
    # localhost:3000 or localhost:5173 (Vite default) to call this API.
    # In production, replace these origins with your actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus metrics endpoint — scraped by your monitoring system."""
        content, content_type = metrics_response()
        return Response(content=content, media_type=content_type)
        
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root URL to the interactive API docs."""
        return RedirectResponse(url="/docs")

    return app


# Module-level `app` instance used by Uvicorn when you run:
#   uvicorn src.api.app:app --reload
app = create_app()