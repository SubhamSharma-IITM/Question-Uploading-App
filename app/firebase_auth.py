"""Firebase Authentication — token verification and session management.

Uses firebase-admin SDK to verify ID tokens from client-side Firebase Auth.
Session cookies (signed, httponly) provide persistence after initial login.
User info (name, role) stored server-side in session — no client storage needed.
"""

import hashlib
import hmac
import time
import logging
from typing import Optional, Dict

from app.config import settings

log = logging.getLogger(__name__)

# ── Session cookie config ──
SESSION_COOKIE_NAME = "acadza_session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 days

# ── User registry: phone -> {name, role} ──
# Add team members here. Phone format: "+91XXXXXXXXXX"
USER_REGISTRY: Dict[str, dict] = {
    "+919661172783": {"name": "Shubham Sharma", "role": "Architect"},
    "+917488290347": {"name": "Team Member", "role": "Developer"},
    # Add more team members as needed:
    # "+91XXXXXXXXXX": {"name": "Full Name", "role": "Team"},
}


def get_user_info(phone: str) -> dict:
    """Look up user name and role from phone number."""
    entry = USER_REGISTRY.get(phone, {})
    return {
        "name": entry.get("name", "User"),
        "role": entry.get("role", "Team"),
    }


# ── In-memory session store ──
# Maps session_token -> {phone, uid, name, role, verified_at, expires_at}
_sessions: Dict[str, dict] = {}


def _get_secret_key() -> str:
    """Derive a signing key from Firebase API key (always available)."""
    return settings.firebase_api_key or "acadza-fallback-key-change-me"


def _sign_token(data: str) -> str:
    """HMAC-sign a token string."""
    return hmac.new(_get_secret_key().encode(), data.encode(), hashlib.sha256).hexdigest()


def create_session(phone: str, uid: str) -> str:
    """Create a session after Firebase token verification. Returns session token."""
    raw = f"{uid}:{phone}:{time.time()}"
    token = hashlib.sha256(raw.encode()).hexdigest()
    sig = _sign_token(token)
    session_token = f"{token}.{sig}"

    user_info = get_user_info(phone)

    _sessions[session_token] = {
        "phone": phone,
        "uid": uid,
        "name": user_info["name"],
        "role": user_info["role"],
        "verified_at": time.time(),
        "expires_at": time.time() + SESSION_MAX_AGE,
    }
    return session_token


def verify_session(session_token: Optional[str]) -> Optional[dict]:
    """Verify a session token. Returns session data or None."""
    if not session_token:
        return None

    parts = session_token.split(".")
    if len(parts) != 2:
        return None

    token, sig = parts
    expected_sig = _sign_token(token)
    if not hmac.compare_digest(sig, expected_sig):
        return None

    session_data = _sessions.get(session_token)
    if not session_data:
        return None

    if time.time() > session_data["expires_at"]:
        _sessions.pop(session_token, None)
        return None

    return session_data


def delete_session(session_token: str):
    """Remove a session."""
    _sessions.pop(session_token, None)


async def verify_firebase_token(id_token: str) -> Optional[dict]:
    """Verify a Firebase ID token server-side.

    Uses firebase-admin if available, otherwise falls back to
    Google's public keys endpoint for stateless verification.
    """
    try:
        import firebase_admin
        from firebase_admin import auth as fb_auth

        # Initialize app if not already done
        if not firebase_admin._apps:
            firebase_admin.initialize_app()

        decoded = fb_auth.verify_id_token(id_token)
        return {
            "uid": decoded["uid"],
            "phone": decoded.get("phone_number", ""),
        }
    except ImportError:
        # firebase-admin not installed -- use REST verification
        log.warning("firebase-admin not installed, using REST token verification")
        return await _verify_token_rest(id_token)
    except Exception as e:
        log.error(f"Firebase token verification failed: {e}")
        return None


async def _verify_token_rest(id_token: str) -> Optional[dict]:
    """Verify Firebase ID token via Google's tokeninfo endpoint (no SDK needed)."""
    import aiohttp

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={settings.firebase_api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"idToken": id_token}) as resp:
            if resp.status != 200:
                log.error(f"Firebase token lookup failed: {resp.status}")
                return None
            data = await resp.json()
            users = data.get("users", [])
            if not users:
                return None
            user = users[0]
            return {
                "uid": user.get("localId", ""),
                "phone": user.get("phoneNumber", ""),
            }
