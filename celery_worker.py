# celery_worker.py  ← NEW FILE at project root
"""
Entry point for the Celery worker process.

Run with:
    celery -A celery_worker.celery_app worker --loglevel=info --concurrency=2

--concurrency=2 means two parallel worker processes. On a CPU-only machine,
this allows two videos to be processed simultaneously. Setting this higher
than your CPU core count gives diminishing returns for CPU-bound work.
"""
from src.api.celery_app import celery_app   # noqa: F401 — imported for the worker

if __name__ == "__main__":
    celery_app.start()