"""Acadza API JWT Authentication Manager.

Centralized token lifecycle management for all API calls to api.acadza.in.
Handles login, caching, validation, and automatic refresh on 401.

Usage:
    from app.pipeline.auth import get_auth_headers, ensure_valid_token

    # Get headers with valid Bearer token for any API call
    headers = get_auth_headers()
    resp = requests.post(url, headers=headers, json=payload)

    # If API returns 401, refresh and retry
    if resp.status_code == 401:
        headers = get_auth_headers(force_refresh=True)
        resp = requests.post(url, headers=headers, json=payload)

Token lifecycle:
- Login: POST /user/login with username/password from .env
- Cache: Token cached in memory + disk (data/auth_token.json)
- Validate: Check expiry from JWT payload before each use
- Refresh: Auto re-login on expiry or 401 response
- Multiple tokens coexist (backend uses stateless JWT verification)
"""

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

_TOKEN_CACHE_FILENAME = "auth_token.json"
_LOGIN_URL = "https://api.acadza.in/user/login"

# Token refresh buffer: refresh 1 hour before actual expiry
_EXPIRY_BUFFER_S = 3600

# Thread lock for concurrent token refresh
_lock = threading.Lock()

# In-memory cache
_cached_token: Optional[str] = None
_cached_expiry: float = 0.0


def _get_credentials() -> Tuple[str, str]:
    """Read API credentials from config/env."""
    from app.config import settings
    username = getattr(settings, "acadza_api_username", "") or ""
    password = getattr(settings, "acadza_api_password", "") or ""
    if not username or not password:
        raise RuntimeError(
            "Acadza API credentials not configured. "
            "Set ACADZA_API_USERNAME and ACADZA_API_PASSWORD in .env"
        )
    return username, password


def _get_cache_path() -> Path:
    """Get disk cache path for token persistence."""
    from app.config import settings
    return Path(settings.data_dir) / _TOKEN_CACHE_FILENAME


def _decode_jwt_expiry(token: str) -> float:
    """Extract expiry timestamp from JWT payload without verification.

    Returns exp as Unix timestamp, or 0.0 if decode fails.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return 0.0
        # Pad base64 if needed
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return float(payload.get("exp", 0))
    except Exception:
        return 0.0


def _is_token_valid(token: str, expiry: float) -> bool:
    """Check if token exists and hasn't expired (with buffer)."""
    if not token:
        return False
    now = time.time()
    return now < (expiry - _EXPIRY_BUFFER_S)


def _load_from_disk() -> Tuple[Optional[str], float]:
    """Load cached token from disk."""
    cache_path = _get_cache_path()
    if not cache_path.exists():
        return None, 0.0
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        token = data.get("token", "")
        expiry = float(data.get("expiry", 0))
        return token, expiry
    except Exception:
        return None, 0.0


def _save_to_disk(token: str, expiry: float) -> None:
    """Persist token to disk for survival across server restarts."""
    cache_path = _get_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"token": token, "expiry": expiry}, f)
    except Exception:
        pass  # Disk cache is optional; in-memory is primary


def _login(logger=None) -> Tuple[str, float]:
    """Perform login and return (token, expiry).

    Args:
        logger: Optional callable(level, message) for log output.

    Raises:
        RuntimeError: If login fails.
    """
    import requests

    username, password = _get_credentials()

    if logger:
        logger("info", "Auth: Logging in to Acadza API...")

    try:
        resp = requests.post(
            _LOGIN_URL,
            headers={"Content-Type": "application/json"},
            json={"username": username, "password": password},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        msg = f"Auth: Login request failed -- {e}"
        if logger:
            logger("error", msg)
        raise RuntimeError(msg)

    if not data.get("isSuccessful"):
        msg = f"Auth: Login rejected -- {data.get('message', 'unknown error')}"
        if logger:
            logger("error", msg)
        raise RuntimeError(msg)

    token = data.get("token", "")
    if not token:
        msg = "Auth: Login succeeded but no token in response"
        if logger:
            logger("error", msg)
        raise RuntimeError(msg)

    expiry = _decode_jwt_expiry(token)
    role = data.get("user", {}).get("role", "unknown")

    if logger:
        import datetime
        exp_str = datetime.datetime.fromtimestamp(
            expiry, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        logger("success", f"Auth: Token acquired (role={role}, expires={exp_str})")

    return token, expiry


def ensure_valid_token(logger=None, force_refresh: bool = False) -> str:
    """Get a valid JWT token, refreshing if needed.

    Thread-safe. Uses in-memory cache first, then disk cache,
    then performs login if both are stale/missing.

    Args:
        logger: Optional callable(level, message) for stage logging.
        force_refresh: If True, skip cache and force re-login.

    Returns:
        Valid JWT token string.

    Raises:
        RuntimeError: If login fails.
    """
    global _cached_token, _cached_expiry

    with _lock:
        # Check in-memory cache
        if not force_refresh and _is_token_valid(_cached_token, _cached_expiry):
            return _cached_token

        # Check disk cache
        if not force_refresh:
            disk_token, disk_expiry = _load_from_disk()
            if _is_token_valid(disk_token, disk_expiry):
                _cached_token = disk_token
                _cached_expiry = disk_expiry
                if logger:
                    logger("info", "Auth: Using cached token from disk")
                return _cached_token

        # Need fresh login
        token, expiry = _login(logger=logger)
        _cached_token = token
        _cached_expiry = expiry
        _save_to_disk(token, expiry)
        return token


def get_auth_headers(logger=None, force_refresh: bool = False) -> Dict[str, str]:
    """Get HTTP headers with valid Authorization Bearer token.

    This is the primary entry point for all API calls.

    Args:
        logger: Optional callable(level, message) for stage logging.
        force_refresh: Force re-login even if cache is valid.

    Returns:
        Dict with Authorization, Content-Type, Accept headers.
    """
    token = ensure_valid_token(logger=logger, force_refresh=force_refresh)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def authenticated_request(
    method: str,
    url: str,
    logger=None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    **kwargs,
):
    """Make an authenticated HTTP request with automatic 401 retry.

    On 401 response, refreshes token and retries once.
    On other failures, retries with exponential backoff.

    Args:
        method: HTTP method ('get', 'post', 'put', 'delete')
        url: Full URL
        logger: Optional callable(level, message) for logging
        max_retries: Max retry attempts for non-auth failures
        retry_delay: Base delay between retries (seconds)
        **kwargs: Passed to requests.request() (json, files, timeout, etc.)

    Returns:
        requests.Response object

    Raises:
        RuntimeError: If all retries exhausted or auth permanently fails.
    """
    import requests

    auth_refreshed = False

    for attempt in range(max_retries):
        # Get auth headers (merge with any existing headers in kwargs)
        auth_headers = get_auth_headers(logger=logger)
        req_headers = {**auth_headers}

        # For file uploads, don't set Content-Type (let requests set multipart boundary)
        if "files" in kwargs:
            req_headers.pop("Content-Type", None)

        # Merge with caller-provided headers
        if "headers" in kwargs:
            caller_headers = kwargs.pop("headers")
            req_headers.update(caller_headers)

        try:
            resp = requests.request(
                method, url,
                headers=req_headers,
                **kwargs,
            )

            # Auth failure -- refresh token and retry once
            if resp.status_code == 401 and not auth_refreshed:
                if logger:
                    logger("warning", "Auth: Got 401 -- refreshing token...")
                auth_refreshed = True
                # Force refresh and retry immediately
                ensure_valid_token(logger=logger, force_refresh=True)
                continue

            return resp

        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                if logger:
                    logger("warning",
                           f"Request failed (attempt {attempt+1}/{max_retries}): {e} "
                           f"-- retrying in {wait:.0f}s")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"All {max_retries} attempts failed for {method.upper()} {url}")


def invalidate_cache() -> None:
    """Clear both in-memory and disk token cache.

    Use this when you know the token is invalid (e.g., backend revoked it).
    """
    global _cached_token, _cached_expiry
    with _lock:
        _cached_token = None
        _cached_expiry = 0.0
        cache_path = _get_cache_path()
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass
