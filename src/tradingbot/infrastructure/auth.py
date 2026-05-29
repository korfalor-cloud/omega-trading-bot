"""Authentication & Authorization — API keys, JWT, RBAC, sessions, rate limiting.

Implements:
- API key management (generate, revoke, list)
- Key encryption using Fernet (cryptography library) or fallback to base64+hashlib
- JWT token generation and validation (manual HMAC-SHA256)
- Role-based access control (admin, trader, viewer)
- Session management with expiry
- Password hashing with PBKDF2 via hashlib
- Rate limiting per user/API key
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PBKDF2_ITERATIONS = 310_000  # OWASP 2024 recommendation for SHA-256
SESSION_DEFAULT_TTL = 3600  # 1 hour
API_KEY_DEFAULT_TTL = 90 * 86400  # 90 days


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _get_fernet():
    """Try to import Fernet from the cryptography library."""
    try:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]
        return Fernet
    except ImportError:
        return None


FernetCls = _get_fernet()


class Encryptor:
    """Symmetric encryption using Fernet when available, else AES-less base64 fallback.

    The fallback is **not** production-grade encryption; it exists so the
    module remains importable without the *cryptography* package.
    """

    def __init__(self, key: Optional[bytes] = None):
        if FernetCls is not None:
            self._fernet = FernetCls(key or FernetCls.generate_key())
            self._mode = "fernet"
        else:
            self._key = key or os.urandom(32)
            self._mode = "fallback"
        logger.info("Encryptor initialised (mode=%s)", self._mode)

    # -- public API ---------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """Return a base64-encoded ciphertext string."""
        if self._mode == "fernet":
            return self._fernet.encrypt(plaintext.encode()).decode()
        # Fallback: XOR with key + base64
        data = plaintext.encode()
        key = self._key
        ct = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
        return base64.urlsafe_b64encode(ct).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string produced by *encrypt*."""
        if self._mode == "fernet":
            return self._fernet.decrypt(ciphertext.encode()).decode()
        ct = base64.urlsafe_b64decode(ciphertext.encode())
        key = self._key
        pt = bytes(b ^ key[i % len(key)] for i, b in enumerate(ct))
        return pt.decode()

    @staticmethod
    def generate_key() -> bytes:
        """Generate a fresh Fernet-compatible key (or random bytes for fallback)."""
        if FernetCls is not None:
            return FernetCls.generate_key()
        return base64.urlsafe_b64encode(os.urandom(32))


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class PasswordHasher:
    """PBKDF2-HMAC-SHA256 password hashing (no external dependency)."""

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = PBKDF2_ITERATIONS) -> str:
        """Return ``salt$iterations$hash`` (all base64)."""
        salt = salt or os.urandom(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return (
            f"{base64.b64encode(salt).decode()}"
            f"${iterations}"
            f"${base64.b64encode(dk).decode()}"
        )

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        """Check *password* against a hash produced by *hash_password*."""
        try:
            salt_b64, iter_str, hash_b64 = stored.split("$")
            salt = base64.b64decode(salt_b64)
            iterations = int(iter_str)
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
            return hmac.compare_digest(dk, base64.b64decode(hash_b64))
        except (ValueError, KeyError):
            return False


# ---------------------------------------------------------------------------
# Roles & Permissions
# ---------------------------------------------------------------------------

class Role(Enum):
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"


# Permissions are additive: each role inherits all permissions of lower roles.
# Hierarchy from lowest to highest: VIEWER < TRADER < ADMIN
_ROLE_HIERARCHY: list[Role] = [Role.VIEWER, Role.TRADER, Role.ADMIN]
_ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: {"read_market_data", "view_positions", "view_reports"},
    Role.TRADER: {"place_order", "cancel_order", "modify_strategy", "view_logs"},
    Role.ADMIN: {"manage_users", "manage_keys", "system_config", "audit_logs"},
}


def permissions_for(role: Role) -> set[str]:
    """Return the full set of permissions for *role* including inherited ones."""
    perms: set[str] = set()
    for r in _ROLE_HIERARCHY:
        perms |= _ROLE_PERMISSIONS.get(r, set())
        if r == role:
            break
    return perms


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class User:
    """A system user."""
    user_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    username: str = ""
    password_hash: str = ""
    role: Role = Role.VIEWER
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class APIKey:
    """An API key record."""
    key_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = ""
    key_hash: str = ""  # SHA-256 of the raw key
    name: str = ""
    role: Role = Role.VIEWER
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    last_used: float = 0.0
    scopes: list[str] = field(default_factory=list)


@dataclass
class Session:
    """An authenticated session."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = ""
    role: Role = Role.VIEWER
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    ip_address: str = ""
    user_agent: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


# ---------------------------------------------------------------------------
# JWT helpers (manual HMAC-SHA256, no external dependency)
# ---------------------------------------------------------------------------

class JWTService:
    """Lightweight JWT implementation using HMAC-SHA256."""

    def __init__(self, secret: str, algorithm: str = "HS256", issuer: str = "omega-trading"):
        self._secret = secret.encode()
        self._algorithm = algorithm
        self._issuer = issuer

    # -- encoding -----------------------------------------------------------

    def encode(self, payload: dict, ttl: int = SESSION_DEFAULT_TTL) -> str:
        """Create a JWT string valid for *ttl* seconds."""
        now = int(time.time())
        header = {"alg": self._algorithm, "typ": "JWT"}
        claims = {
            **payload,
            "iss": self._issuer,
            "iat": now,
            "exp": now + ttl,
            "jti": uuid.uuid4().hex,
        }
        segments = [
            self._b64json(header),
            self._b64json(claims),
        ]
        signing_input = ".".join(segments).encode()
        sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        segments.append(self._b64url(sig))
        return ".".join(segments)

    # -- decoding / validation ----------------------------------------------

    def decode(self, token: str) -> dict:
        """Validate and decode a JWT. Raises *ValueError* on failure."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed JWT")
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        actual = self._b64url_decode(sig_b64)
        if not hmac.compare_digest(expected, actual):
            raise ValueError("Invalid JWT signature")
        payload = json.loads(self._b64url_decode(payload_b64))
        if int(time.time()) > payload.get("exp", 0):
            raise ValueError("JWT expired")
        return payload

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _b64json(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).rstrip(b"=").decode()

    @staticmethod
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        s += "=" * (4 - len(s) % 4)
        return base64.urlsafe_b64decode(s)


# ---------------------------------------------------------------------------
# Per-key / per-user rate limiter (token bucket)
# ---------------------------------------------------------------------------

@dataclass
class _Bucket:
    tokens: float
    capacity: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)

    def acquire(self, n: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class AuthRateLimiter:
    """Per-identifier token-bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self._rpm = requests_per_minute
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, identifier: str) -> bool:
        """Return True if the request is allowed under the rate limit."""
        bucket = self._buckets.get(identifier)
        if bucket is None:
            bucket = _Bucket(
                tokens=float(self._rpm),
                capacity=float(self._rpm),
                refill_rate=self._rpm / 60.0,
            )
            self._buckets[identifier] = bucket
        return bucket.acquire()

    def reset(self, identifier: str) -> None:
        """Reset the bucket for *identifier* (e.g. after role upgrade)."""
        self._buckets.pop(identifier, None)


# ---------------------------------------------------------------------------
# JWT payload construction helper
# ---------------------------------------------------------------------------

def _jwt_payload(user_id: str, role: Role, extra: Optional[dict] = None) -> dict:
    """Build a standard JWT claims dict."""
    claims: dict = {
        "sub": user_id,
        "role": role.value,
        "perms": sorted(permissions_for(role)),
    }
    if extra:
        claims.update(extra)
    return claims


# ---------------------------------------------------------------------------
# Main AuthManager
# ---------------------------------------------------------------------------

class AuthManager:
    """Central authentication and authorisation service.

    Manages users, API keys, sessions, JWTs, and rate limits.
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        requests_per_minute: int = 60,
    ):
        self._jwt = JWTService(secret=jwt_secret or secrets.token_hex(32))
        self._encryptor = Encryptor(encryption_key)
        self._hasher = PasswordHasher()
        self._rate_limiter = AuthRateLimiter(requests_per_minute)

        # In-memory stores (production would back these with a database)
        self._users: dict[str, User] = {}
        self._api_keys: dict[str, APIKey] = {}
        self._sessions: dict[str, Session] = {}

        logger.info("AuthManager initialised")

    # -- properties ---------------------------------------------------------

    @property
    def jwt_service(self) -> JWTService:
        return self._jwt

    @property
    def encryptor(self) -> Encryptor:
        return self._encryptor

    # -- User management ----------------------------------------------------

    def create_user(self, username: str, password: str, role: Role = Role.VIEWER) -> User:
        """Register a new user. Raises *ValueError* if username is taken."""
        for u in self._users.values():
            if u.username == username:
                raise ValueError(f"Username '{username}' already exists")
        user = User(
            username=username,
            password_hash=self._hasher.hash_password(password),
            role=role,
        )
        self._users[user.user_id] = user
        logger.info("User created: %s (role=%s)", username, role.value)
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Verify credentials and return the user, or None."""
        for user in self._users.values():
            if user.username == username and user.is_active:
                if self._hasher.verify_password(password, user.password_hash):
                    user.last_login = time.time()
                    logger.info("User authenticated: %s", username)
                    return user
                logger.warning("Failed login attempt for user: %s", username)
                return None
        return None

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return list(self._users.values())

    def deactivate_user(self, user_id: str) -> bool:
        user = self._users.get(user_id)
        if user:
            user.is_active = False
            # Revoke all sessions and keys for this user
            for sid, s in list(self._sessions.items()):
                if s.user_id == user_id:
                    del self._sessions[sid]
            for kid, k in list(self._api_keys.items()):
                if k.user_id == user_id:
                    k.is_active = False
            logger.info("User deactivated: %s", user_id)
            return True
        return False

    def change_user_role(self, user_id: str, new_role: Role) -> bool:
        user = self._users.get(user_id)
        if user:
            user.role = new_role
            logger.info("User %s role changed to %s", user_id, new_role.value)
            return True
        return False

    # -- API Key management -------------------------------------------------

    def generate_api_key(
        self,
        user_id: str,
        name: str = "",
        role: Optional[Role] = None,
        ttl: int = API_KEY_DEFAULT_TTL,
        scopes: Optional[list[str]] = None,
    ) -> tuple[str, APIKey]:
        """Generate a new API key. Returns (raw_key, api_key_record).

        The raw key is shown **once**; only the hash is stored.
        """
        user = self._users.get(user_id)
        if not user or not user.is_active:
            raise ValueError(f"Invalid or inactive user: {user_id}")

        raw_key = f"omega_{secrets.token_urlsafe(32)}"
        key_record = APIKey(
            user_id=user_id,
            key_hash=hashlib.sha256(raw_key.encode()).hexdigest(),
            name=name or f"key-{uuid.uuid4().hex[:8]}",
            role=role or user.role,
            expires_at=time.time() + ttl,
            scopes=scopes or [],
        )
        self._api_keys[key_record.key_id] = key_record
        logger.info("API key generated for user %s: %s", user_id, key_record.key_id)
        return raw_key, key_record

    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate a raw API key and return its record, or None."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        for key in self._api_keys.values():
            if key.key_hash == key_hash and key.is_active:
                if time.time() > key.expires_at > 0:
                    logger.warning("Expired API key used: %s", key.key_id)
                    return None
                key.last_used = time.time()
                return key
        return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke (deactivate) an API key."""
        key = self._api_keys.get(key_id)
        if key:
            key.is_active = False
            logger.info("API key revoked: %s", key_id)
            return True
        return False

    def list_api_keys(self, user_id: str) -> list[APIKey]:
        """List all API keys for a user (including revoked ones)."""
        return [k for k in self._api_keys.values() if k.user_id == user_id]

    def rotate_api_key(self, key_id: str, ttl: int = API_KEY_DEFAULT_TTL) -> Optional[tuple[str, APIKey]]:
        """Revoke old key and issue a new one with the same settings."""
        old = self._api_keys.get(key_id)
        if not old:
            return None
        old.is_active = False
        return self.generate_api_key(
            user_id=old.user_id,
            name=old.name,
            role=old.role,
            ttl=ttl,
            scopes=old.scopes,
        )

    # -- Session management -------------------------------------------------

    def create_session(
        self,
        user_id: str,
        ttl: int = SESSION_DEFAULT_TTL,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Session:
        """Create an authenticated session."""
        user = self._users.get(user_id)
        if not user:
            raise ValueError(f"Unknown user: {user_id}")
        session = Session(
            user_id=user_id,
            role=user.role,
            expires_at=time.time() + ttl,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self._sessions[session.session_id] = session
        logger.info("Session created for user %s: %s", user_id, session.session_id)
        return session

    def validate_session(self, session_id: str) -> Optional[Session]:
        """Return the session if valid and not expired."""
        session = self._sessions.get(session_id)
        if session and not session.is_expired:
            return session
        if session and session.is_expired:
            del self._sessions[session_id]
        return None

    def revoke_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info("Session revoked: %s", session_id)
            return True
        return False

    def revoke_all_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user. Returns count of revoked sessions."""
        to_remove = [sid for sid, s in self._sessions.items() if s.user_id == user_id]
        for sid in to_remove:
            del self._sessions[sid]
        logger.info("Revoked %d sessions for user %s", len(to_remove), user_id)
        return len(to_remove)

    def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    # -- JWT tokens ---------------------------------------------------------

    def issue_token(self, user_id: str, role: Role, ttl: int = SESSION_DEFAULT_TTL) -> str:
        """Issue a signed JWT for *user_id* with *role*."""
        payload = _jwt_payload(user_id, role)
        return self._jwt.encode(payload, ttl=ttl)

    def verify_token(self, token: str) -> Optional[dict]:
        """Validate a JWT and return its payload, or None."""
        try:
            return self._jwt.decode(token)
        except ValueError as exc:
            logger.warning("JWT verification failed: %s", exc)
            return None

    # -- Rate limiting ------------------------------------------------------

    def check_rate_limit(self, identifier: str) -> bool:
        """Return True if the request is within the rate limit."""
        return self._rate_limiter.allow(identifier)

    # -- Authorisation helpers ----------------------------------------------

    def authorize(self, token_or_key: str, required_permission: str) -> bool:
        """Check if the bearer of *token_or_key* has *required_permission*.

        Accepts either a JWT token or a raw API key.
        """
        # Rate limit check
        identifier = self._extract_identifier(token_or_key)
        if not self._rate_limiter.allow(identifier):
            logger.warning("Rate limit exceeded for %s", identifier[:16])
            return False

        role = self._resolve_role(token_or_key)
        if role is None:
            return False
        return required_permission in permissions_for(role)

    def _resolve_role(self, token_or_key: str) -> Optional[Role]:
        """Try to resolve a Role from a JWT or API key."""
        # Try JWT first
        try:
            payload = self._jwt.decode(token_or_key)
            return Role(payload.get("role", "viewer"))
        except ValueError:
            pass
        # Try API key
        api_key = self.validate_api_key(token_or_key)
        if api_key:
            return api_key.role
        return None

    def _extract_identifier(self, token_or_key: str) -> str:
        """Best-effort identifier for rate limiting."""
        try:
            payload = self._jwt.decode(token_or_key)
            return payload.get("sub", token_or_key[:32])
        except ValueError:
            return hashlib.sha256(token_or_key.encode()).hexdigest()[:32]
