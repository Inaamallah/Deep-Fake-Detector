# src/api/database.py  ← NEW FILE
"""
SQLite-backed job store.

Why SQLite and not PostgreSQL?
For a single-machine deployment, SQLite is ideal — zero infrastructure,
ACID-compliant, and fast enough for thousands of jobs per day. If you later
need to scale to multiple API servers, swapping to PostgreSQL is a one-day
task (change the connection string and swap sqlite3 for psycopg2).

Thread safety note: we open a new connection per operation using the
context manager rather than sharing one connection across threads. SQLite
in WAL mode handles concurrent reads well and serialises writes automatically.
`check_same_thread=False` is safe here because each call creates its own
connection object — we are not sharing a single connection across threads.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from config import settings
from src.api.models import JobStatus

# Store the database file alongside the other data directories
DB_PATH = settings.data_dir / "jobs.db"


@contextmanager
def _db_conn() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that opens a connection, yields it, commits on success,
    and rolls back + closes on any exception. Always closes the connection.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row   # rows behave like dicts — row["col"]
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the jobs table if it does not already exist.
    Called once at API startup via the FastAPI lifespan event.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    with _db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id        TEXT PRIMARY KEY,
                status        TEXT NOT NULL DEFAULT 'PENDING',
                source        TEXT,
                video_id      TEXT,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL,
                error_message TEXT,
                result_path   TEXT
            )
        """)


def create_job(source: str) -> str:
    """
    Insert a new job row with PENDING status. Returns the generated job_id.

    We use uuid4 (random UUID) rather than an auto-increment integer because:
      - It is globally unique — safe if you later shard across databases.
      - It is unpredictable — users cannot enumerate other users' jobs by
        guessing sequential IDs.
    """
    job_id = str(uuid.uuid4())
    now    = time.time()
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO jobs
               (job_id, status, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, JobStatus.PENDING.value, source, now, now),
        )
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch one job by ID. Returns a plain dict or None if not found.
    We return a dict (not a Pydantic model) so the database layer has
    no dependency on API models — keeping the layers independent.
    """
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def update_job(
    job_id:        str,
    status:        JobStatus,
    video_id:      Optional[str] = None,
    error_message: Optional[str] = None,
    result_path:   Optional[str] = None,
) -> None:
    """
    Update the status and optional fields of a job.

    The COALESCE pattern in the SQL means: "use the new value if provided,
    otherwise keep whatever is already in the column." This lets callers
    update only the fields they care about without overwriting everything.

    For example, when transitioning from DOWNLOADING to EXTRACTING_FRAMES
    we pass the video_id (now known) but leave result_path as None — and
    any previously stored result_path stays untouched.
    """
    now = time.time()
    with _db_conn() as conn:
        conn.execute(
            """UPDATE jobs
               SET status        = ?,
                   video_id      = COALESCE(?, video_id),
                   error_message = COALESCE(?, error_message),
                   result_path   = COALESCE(?, result_path),
                   updated_at    = ?
               WHERE job_id = ?""",
            (
                status.value,
                video_id,
                error_message,
                result_path,
                now,
                job_id,
            ),
        )