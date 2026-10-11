import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def auth_disabled() -> bool:
    return os.getenv("ALLOW_NO_AUTH", "").lower() == "true"


def check_auth_config() -> None:
    """Refuse to run without an API key unless auth was explicitly disabled for local dev."""
    if not os.getenv("API_KEY") and not auth_disabled():
        raise RuntimeError("API_KEY is not set. Set it, or set ALLOW_NO_AUTH=true for local development only.")


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        if auth_disabled():
            return
        raise HTTPException(500, "Server misconfigured: API_KEY is not set")
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "APIKey"})
