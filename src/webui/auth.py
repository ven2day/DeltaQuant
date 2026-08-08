"""Authentication for the web dashboard: password hashing, session tokens, and
login-attempt rate limiting.

The dashboard exposes wallet balance, positions, and trading activity over the
network (see M-8, DeltaQuant-Quant-Risk-Review.md: "API protection relies mainly on
localhost binding... any remote binding... would require authentication"). This
module is that authentication layer. It deliberately avoids adding new third-party
crypto dependencies (bcrypt/passlib/PyJWT) -- everything here is stdlib `hashlib`,
`hmac`, and `secrets`, which are sufficient and audit-friendly for a single-operator
password + signed session token:

- Passwords are hashed with PBKDF2-HMAC-SHA256, a random per-password salt, and a
  600,000-iteration count (OWASP's 2023 minimum recommendation for PBKDF2-SHA256).
  Verification uses `hmac.compare_digest` (constant-time) to avoid timing attacks.
- Sessions are a stateless HMAC-signed token (subject + expiry, base64url-encoded,
  signed with a server-side secret) -- no server-side session store to leak or lose
  on restart, and forging one without the secret is infeasible.
- Login attempts are rate-limited per client IP with a sliding-window lockout, so a
  network attacker cannot brute-force the password even if it's weak.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 600_000
_SALT_BYTES = 16


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash a password for storage in WEB_UI_PASSWORD_HASH.

    Format: ``pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>`` -- iterations and
    salt travel with the hash so verification never has to guess parameters, and the
    iteration count can be raised later without invalidating already-stored hashes
    from a lower count (each hash is self-describing).
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return (
        f"{PBKDF2_ALGORITHM}${iterations}$"
        f"{base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"
    )


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time verification against a hash_password() string.

    Returns False (never raises) for a malformed/unrecognized stored_hash -- an
    operator who hasn't run set_dashboard_password.py yet, or a corrupted .env
    value, must fail closed to "wrong password", not crash the login route.
    """
    try:
        algorithm, iterations_s, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != PBKDF2_ALGORITHM:
            return False
        iterations = int(iterations_s)
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(digest_b64)
    except (ValueError, TypeError):
        logger.warning("WEB_UI_PASSWORD_HASH is malformed; rejecting all logins")
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_session_token(subject: str, *, secret: str, ttl_minutes: int) -> str:
    """A stateless, tamper-evident session token: ``<payload_b64>.<signature_b64>``.

    The payload (subject + expiry) is not encrypted, only signed -- it must not
    carry anything more sensitive than a username, which is already known to
    whoever is presenting valid credentials for it.
    """
    payload = {"sub": subject, "exp": int(time.time()) + ttl_minutes * 60}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256)
    return f"{payload_b64}.{_b64url_encode(signature.digest())}"


def verify_session_token(token: str, *, secret: str) -> str | None:
    """Verify signature and expiry; return the subject, or None if invalid/expired.

    Never raises -- any malformed token (wrong shape, bad base64, tampered
    signature, expired) is treated identically as "not authenticated".
    """
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        expected_sig = hmac.new(
            secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).digest()
        actual_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if not isinstance(payload, dict):
            return None
        exp = payload.get("exp")
        sub = payload.get("sub")
        if not isinstance(exp, int) or not isinstance(sub, str):
            return None
        if time.time() >= exp:
            return None
        return sub
    except Exception:
        return None


@dataclass
class LoginAttemptTracker:
    """Sliding-window lockout per client IP, to blunt online password guessing.

    In-memory and per-process -- acceptable here because the web UI is one
    process (see WebUIServer's docstring: runs in the same asyncio loop as the
    trading loop, not a separate scaled-out service). Resets on restart, which
    is a reasonable tradeoff for a single-operator dashboard rather than pulling
    in a shared store for this alone.
    """

    max_attempts: int = 5
    lockout_minutes: int = 15
    _failures: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def _prune(self, ip: str, now: float) -> None:
        window_start = now - self.lockout_minutes * 60
        bucket = self._failures[ip]
        while bucket and bucket[0] < window_start:
            bucket.popleft()

    def is_locked(self, ip: str) -> bool:
        now = time.time()
        self._prune(ip, now)
        return len(self._failures[ip]) >= self.max_attempts

    def record_failure(self, ip: str) -> None:
        now = time.time()
        self._prune(ip, now)
        self._failures[ip].append(now)

    def reset(self, ip: str) -> None:
        self._failures.pop(ip, None)
