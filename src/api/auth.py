# src/api/auth.py  ← NEW FILE
"""
API key authentication using FastAPI's dependency injection system.

How it works: any route that includes `Depends(verify_api_key)` in its
signature will have FastAPI automatically call verify_api_key() before
the route function runs. If the key is invalid, FastAPI short-circuits
and returns 403 without ever calling your route function.

This is far cleaner than manually checking headers inside every route,
and it shows up automatically in the generated OpenAPI docs.
"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import settings

# auto_error=False means FastAPI will NOT automatically return 403 when
# the header is absent. We handle the missing-key case ourselves below,
# which lets us return a more descriptive error message than the default.
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    """
    FastAPI dependency. Validates the X-API-Key header.

    Returns the API key string on success (useful if you later want
    to log which key was used for audit trails).
    Raises HTTP 403 on missing or invalid key.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "API key is missing. "
                "Include it in the X-API-Key request header."
            ),
        )

    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key