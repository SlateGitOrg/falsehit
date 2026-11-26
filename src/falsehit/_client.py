"""The in-process test client, imported without the httpx deprecation notice.

Recent Starlette releases warn when `TestClient` runs on `httpx` instead of `httpx2`. The client
works the same either way, so the notice is silenced at import time only.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

__all__ = ["TestClient"]
