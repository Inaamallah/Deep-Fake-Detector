# src/api/limiter.py  ← NEW FILE
from slowapi import Limiter
from slowapi.util import get_remote_address

# get_remote_address extracts the client's IP from the request.
# This is the "key" used to track how many requests each IP has made.
# In production behind a reverse proxy (Nginx), you would use
# get_remote_address configured to trust X-Forwarded-For headers.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],  # global fallback — routes can override
)