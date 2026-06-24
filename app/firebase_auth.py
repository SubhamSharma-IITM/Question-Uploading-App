"""PIN-based Authentication — phone + 6-digit PIN with rate limiting.

Server-side PIN verification with SHA-256 hashed storage.
Session cookies (signed, httponly) provide persistence after login.
3 failed attempts = 15-minute lockout per phone number.
"""

import hashlib
import hmac
import time
import logging
from typing import Optional, Dict

log = logging.getLogger(__name__)

# ── Session cookie config ──
SESSION_COOKIE_NAME = "acadza_session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 days

# ── Rate limiting ──
MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 15 * 60  # 15 minutes
# {phone: {"count": int, "locked_until": float}}
_rate_limits: Dict[str, dict] = {}


def _hash_pin(pin: str) -> str:
    """SHA-256 hash a PIN."""
    return hashlib.sha256(pin.encode()).hexdigest()


# ── User registry: phone -> {name, role, pin_hash} ──
# PINs are SHA-256 hashed. To generate a hash:
#   python -c "import hashlib; print(hashlib.sha256('123456'.encode()).hexdigest())"
# Default PIN for new accounts: 143143 (change on first login -- TODO)
USER_REGISTRY: Dict[str, dict] = {
    # ── Admins ──
    "+919661172783": {
        "name": "Subham Sharma",
        "role": "Admin",
        "pin_hash": _hash_pin("143143"),
    },
    "+917992279131": {
        "name": "Manish Anand",
        "role": "Admin",
        "pin_hash": _hash_pin("143143"),
    },
    "+917667909991": {
        "name": "Ashish Anand",
        "role": "Admin",
        "pin_hash": _hash_pin("143143"),
    },
    "+919125985931": {
        "name": "Saurabh Agrahari",
        "role": "Admin",
        "pin_hash": _hash_pin("143143"),
    },
    # ── Team Members ──
    "+919279006541": {
        "name": "Khushboo Kumari",
        "role": "Team",
        "pin_hash": _hash_pin("143143"),
    },
    "+918108728834": {
        "name": "Rubina Khan",
        "role": "Team",
        "pin_hash": _hash_pin("143143"),
    },
    "+918452099268": {
        "name": "Rutuja Mayekar",
        "role": "Team",
        "pin_hash": _hash_pin("143143"),
    },
    "+917644045760": {
        "name": "Raushan Kumar",
        "role": "Team",
        "pin_hash": _hash_pin("143143"),
    },
}


def get_user_info(phone: str) -> dict:
    """Look up user name and role from phone number."""
    entry = USER_REGISTRY.get(phone, {})
    return {
        "name": entry.get("name", "User"),
        "role": entry.get("role", "Team"),
    }


def check_rate_limit(phone: str) -> Optional[int]:
    """Check if phone is rate-limited. Returns seconds remaining or None."""
    rl = _rate_limits.get(phone)
    if not rl:
        return None
    if rl.get("locked_until", 0) > time.time():
        return int(rl["locked_until"] - time.time())
    # Lockout expired — reset
    if rl.get("count", 0) >= MAX_ATTEMPTS:
        _rate_limits.pop(phone, None)
    return None


def record_failed_attempt(phone: str) -> int:
    """Record a failed login attempt. Returns attempts remaining."""
    rl = _rate_limits.setdefault(phone, {"count": 0, "locked_until": 0})
    rl["count"] = rl.get("count", 0) + 1
    remaining = MAX_ATTEMPTS - rl["count"]
    if remaining <= 0:
        rl["locked_until"] = time.time() + LOCKOUT_SECONDS
        log.warning(f"Phone {phone[-4:]} locked for {LOCKOUT_SECONDS//60} minutes after {MAX_ATTEMPTS} failed attempts")
    return max(remaining, 0)


def clear_rate_limit(phone: str):
    """Clear rate limit after successful login."""
    _rate_limits.pop(phone, None)


def verify_pin(phone: str, pin: str) -> Optional[dict]:
    """Verify phone + PIN. Returns user info or None."""
    # Normalize phone
    if not phone.startswith("+91"):
        phone = "+91" + phone.lstrip("+")

    user = USER_REGISTRY.get(phone)
    if not user:
        return None

    if _hash_pin(pin) == user["pin_hash"]:
        return {
            "name": user["name"],
            "role": user["role"],
            "phone": phone,
        }
    return None


# ── In-memory session store ──
# Maps session_token -> {phone, uid, name, role, verified_at, expires_at}
_sessions: Dict[str, dict] = {}


def _get_secret_key() -> str:
    """Derive a signing key."""
    return "acadza-pin-auth-secret-key"


def _sign_token(data: str) -> str:
    """HMAC-sign a token string."""
    return hmac.new(_get_secret_key().encode(), data.encode(), hashlib.sha256).hexdigest()


def create_session(phone: str, name: str, role: str) -> str:
    """Create a session after PIN verification. Returns session token."""
    raw = f"{phone}:{name}:{time.time()}"
    token = hashlib.sha256(raw.encode()).hexdigest()
    sig = _sign_token(token)
    session_token = f"{token}.{sig}"

    _sessions[session_token] = {
        "phone": phone,
        "name": name,
        "role": role,
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
