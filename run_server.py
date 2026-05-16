# run_server.py  ← NEW FILE at project root
"""
Entry point for the FastAPI / Uvicorn API server.

Run with:
    python run_server.py

Or directly with Uvicorn for more control:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host     = "127.0.0.1",
        port     = 8000,
        reload   = True,   # auto-reload on code changes — disable in production
        log_level= "info",
    )