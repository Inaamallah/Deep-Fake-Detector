# src/api/celery_app.py  ← NEW FILE
import sys
from celery import Celery
from config import settings

# The `include` list tells Celery where to find task definitions.
# It must be set here — before any task is imported — so that when the
# worker starts, it knows which modules to scan for @celery_app.task decorators.
celery_app = Celery(
    "deepfake_detector",
    broker=settings.celery_broker_url,
    backend=settings.celery_backend_url,
    include=["src.api.tasks"],
)

celery_app.conf.update(
    # Always use JSON for serialisation — never pickle.
    # Pickle is a security risk because it can execute arbitrary Python
    # when deserialised. JSON is safe, readable, and portable.
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    timezone="UTC",
    enable_utc=True,

    # IMPORTANT: task_acks_late=True means the task is acknowledged
    # (removed from the queue) only AFTER it completes, not when it starts.
    # If the worker crashes mid-pipeline, the task stays in the queue and
    # can be picked up by another worker. This is critical for reliability.
    task_acks_late=True,

    # worker_prefetch_multiplier=1 means each worker process fetches only
    # one task at a time. Our tasks are CPU-intensive and long-running —
    # fetching multiple tasks ahead of time would starve other workers.
    worker_prefetch_multiplier=1,

    # Store task state (PENDING, STARTED, SUCCESS, FAILURE) in Redis.
    # This lets us check task progress independently of our own DB.
    task_track_started=True,

    # On Windows, the default 'prefork' pool (billiard) crashes with
    # PermissionError on semaphores. Use 'solo' pool instead, which
    # processes tasks sequentially in the main worker process.
    # On Linux/macOS, prefork works fine — we leave it as default.
    **({"worker_pool": "solo"} if sys.platform == "win32" else {}),
)