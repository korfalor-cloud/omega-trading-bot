"""Security Utilities — input validation, sanitisation, CORS, request signing, audit logging.

Implements:
- Input validation and sanitisation
- SQL injection prevention (parameterised query helpers)
- XSS prevention for API responses
- CORS configuration
- Request signing verification (HMAC-SHA256)
- Secure HTTP headers
- Audit logging for security events
"""
from __future__ import annotations

import hashlib
import html
import hmac
import ipaddress
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

class AuditAction(Enum):
    """Security-relevant actions to audit."""
    LOGIN_SUCCESS = auto()
    LOGIN_FAILURE = auto()
    LOGOUT = auto()
    TOKEN_ISSUED = auto()
    TOKEN_REVOKED = auto()
    API_KEY_GENERATED = auto()
    API_KEY_REVOKED = auto()
    PERMISSION_DENIED = auto()
    RATE_LIMIT_HIT = auto()
    INVALID_INPUT = auto()
    SQL_INJECTION_ATTEMPT = auto()
    XSS_ATTEMPT = auto()
    REQUEST_SIGNATURE_INVALID = auto()
    CONFIG_CHANGED = auto()
    USER_CREATED = auto()
    USER_DEACTIVATED = auto()
    ROLE_CHANGED = auto()
    SESSION_CREATED = auto()
    SESSION_REVOKED = auto()
    CREDENTIAL_ACCESSED = auto()
    CREDENTIAL_ROTATED = auto()
    RECOVERY_STARTED = auto()
    RECOVERY_COMPLETED = auto()
    SHUTDOWN_INITIATED = auto()


@dataclass
class AuditEntry:
    """A single audit log record."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    action: AuditAction = AuditAction.LOGIN_SUCCESS
    actor: str = ""  # user_id or "system"
    source_ip: str = ""
    resource: str = ""
    detail: str = ""
    success: bool = True
    metadata: dict = field(default_factory=dict)


class AuditLogger:
    """Append-only in-memory audit log with structured entries.

    In production this should be backed by a persistent, tamper-evident store.
    """

    def __init__(self, max_entries: int = 50_000):
        self._entries: list[AuditEntry] = []
        self._max = max_entries

    def log(
        self,
        action: AuditAction,
        actor: str = "system",
        source_ip: str = "",
        resource: str = "",
        detail: str = "",
        success: bool = True,
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record an audit event."""
        entry = AuditEntry(
            action=action,
            actor=actor,
            source_ip=source_ip,
            resource=resource,
            detail=detail,
            success=success,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

        log_fn = logger.info if success else logger.warning
        log_fn(
            "AUDIT [%s] actor=%s resource=%s detail=%s",
            action.name,
            actor,
            resource,
            detail,
        )
        return entry

    def query(
        self,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        results = self._entries
        if action is not None:
            results = [e for e in results if e.action == action]
        if actor is not None:
            results = [e for e in results if e.actor == actor]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results[-limit:]

    @property
    def total(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


# ---------------------------------------------------------------------------
# Input validation & sanitisation
# ---------------------------------------------------------------------------

# Pre-compiled patterns for common injection vectors
_SQL_INJECTION_RE = re.compile(
    r"(\b(union|select|insert|update|delete|drop|alter|exec|execute|truncate|declare)\b"
    r"|--|/\*|\*/|;.*\b(select|insert|update|delete|drop)\b"
    r"|'\s*(or|and)\s*'?\d|'\s*or\s+'?\d\s*=\s*'?\d)",
    re.IGNORECASE,
)
_XSS_TAG_RE = re.compile(r"<\s*script|javascript:|on\w+\s*=|data:\s*text/html", re.IGNORECASE)
_SAFE_STRING_RE = re.compile(r"^[a-zA-Z0-9_\-.@+ ]+$")


class InputValidator:
    """Validate and sanitise untrusted input."""

    @staticmethod
    def validate_string(
        value: str,
        *,
        max_length: int = 1024,
        pattern: Optional[str] = None,
        allow_empty: bool = False,
    ) -> str:
        """Validate a string input. Raises *ValueError* on failure."""
        if not allow_empty and not value.strip():
            raise ValueError("Input must not be empty")
        if len(value) > max_length:
            raise ValueError(f"Input exceeds max length of {max_length}")
        if pattern and not re.match(pattern, value):
            raise ValueError(f"Input does not match pattern: {pattern}")
        return value

    @staticmethod
    def validate_numeric(
        value: Any,
        *,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> float:
        """Validate numeric input. Raises *ValueError* on failure."""
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value: {value}") from exc
        if min_val is not None and num < min_val:
            raise ValueError(f"Value {num} below minimum {min_val}")
        if max_val is not None and num > max_val:
            raise ValueError(f"Value {num} above maximum {max_val}")
        return num

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate a trading symbol (e.g. BTC/USDT)."""
        if not re.match(r"^[A-Z0-9]{2,10}/[A-Z0-9]{2,10}$", symbol):
            raise ValueError(f"Invalid trading symbol: {symbol}")
        return symbol

    @staticmethod
    def validate_side(side: str) -> str:
        """Validate order side."""
        side = side.lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side: {side}")
        return side

    @staticmethod
    def validate_ip(ip_str: str) -> str:
        """Validate an IP address string."""
        try:
            ipaddress.ip_address(ip_str)
            return ip_str
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {ip_str}") from exc


# ---------------------------------------------------------------------------
# SQL injection prevention
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Build parameterised SQL queries safely.

    This helper ensures all user-supplied values are passed as query
    parameters rather than interpolated into the SQL string.
    """

    def __init__(self, base_query: str):
        self._query = base_query
        self._params: list[Any] = []

    def where(self, condition: str, *params: Any) -> "QueryBuilder":
        """Append a WHERE clause with parameterised values."""
        if "WHERE" in self._query.upper():
            self._query += f" AND {condition}"
        else:
            self._query += f" WHERE {condition}"
        self._params.extend(params)
        return self

    def order_by(self, column: str, ascending: bool = True) -> "QueryBuilder":
        """Append ORDER BY. Column name is validated to prevent injection."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", column):
            raise ValueError(f"Invalid column name: {column}")
        direction = "ASC" if ascending else "DESC"
        self._query += f" ORDER BY {column} {direction}"
        return self

    def limit(self, count: int, offset: int = 0) -> "QueryBuilder":
        """Append LIMIT/OFFSET."""
        self._query += " LIMIT ? OFFSET ?"
        self._params.extend([count, offset])
        return self

    def build(self) -> tuple[str, list[Any]]:
        """Return (query_string, params_list)."""
        return self._query, list(self._params)


def sanitise_identifier(name: str) -> str:
    """Sanitise a SQL identifier (table or column name).

    Only allows alphanumeric characters and underscores.
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


# ---------------------------------------------------------------------------
# XSS prevention
# ---------------------------------------------------------------------------

class XSSSanitiser:
    """Prevent XSS in API responses."""

    # Tags that are always stripped
    _STRIP_RE = re.compile(
        r"<\s*/?\s*(script|iframe|object|embed|form|input|textarea|button|link|meta|style)\b[^>]*>",
        re.IGNORECASE,
    )
    _EVENT_HANDLER_RE = re.compile(r"\bon\w+\s*=", re.IGNORECASE)
    _JAVASCRIPT_URI_RE = re.compile(r"javascript\s*:", re.IGNORECASE)

    @staticmethod
    def sanitise(value: str) -> str:
        """HTML-escape a string for safe inclusion in responses."""
        return html.escape(value, quote=True)

    @classmethod
    def sanitise_dict(cls, data: dict) -> dict:
        """Recursively sanitise all string values in a dict."""
        out: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[k] = cls.sanitise(v)
            elif isinstance(v, dict):
                out[k] = cls.sanitise_dict(v)
            elif isinstance(v, list):
                out[k] = [cls.sanitise(i) if isinstance(i, str) else i for i in v]
            else:
                out[k] = v
        return out

    @classmethod
    def has_xss(cls, value: str) -> bool:
        """Return True if the value contains potential XSS content."""
        return bool(
            cls._STRIP_RE.search(value)
            or cls._EVENT_HANDLER_RE.search(value)
            or cls._JAVASCRIPT_URI_RE.search(value)
            or _XSS_TAG_RE.search(value)
        )


# ---------------------------------------------------------------------------
# SQL injection detection
# ---------------------------------------------------------------------------

class SQLInjectionDetector:
    """Detect common SQL injection patterns in user input."""

    @staticmethod
    def is_suspicious(value: str) -> bool:
        """Return True if *value* looks like a SQL injection attempt."""
        return bool(_SQL_INJECTION_RE.search(value))

    @classmethod
    def check_dict(cls, data: dict) -> list[str]:
        """Return list of keys whose values look suspicious."""
        return [k for k, v in data.items() if isinstance(v, str) and cls.is_suspicious(v)]


# ---------------------------------------------------------------------------
# Request signing
# ---------------------------------------------------------------------------

class RequestSigner:
    """HMAC-SHA256 request signing and verification.

    Canonical message format: ``METHOD\\nPATH\\nTIMESTAMP\\nBODY_HASH``
    """

    def __init__(self, secret: str):
        self._secret = secret.encode()

    def sign(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        timestamp: Optional[int] = None,
    ) -> dict[str, str]:
        """Sign a request and return headers to attach."""
        ts = str(timestamp or int(time.time()))
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{method.upper()}\n{path}\n{ts}\n{body_hash}"
        signature = hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()
        return {
            "X-Signature": signature,
            "X-Timestamp": ts,
            "X-Body-Hash": body_hash,
        }

    def verify(
        self,
        method: str,
        path: str,
        body: bytes,
        signature: str,
        timestamp: str,
        max_age: int = 300,
    ) -> bool:
        """Verify a signed request. Rejects if older than *max_age* seconds."""
        try:
            ts = int(timestamp)
        except (ValueError, TypeError):
            return False
        if abs(time.time() - ts) > max_age:
            logger.warning("Request signature expired (age=%ds)", int(time.time() - ts))
            return False
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{method.upper()}\n{path}\n{timestamp}\n{body_hash}"
        expected = hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

@dataclass
class CORSConfig:
    """CORS policy configuration."""
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    allowed_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allowed_headers: list[str] = field(default_factory=lambda: ["Content-Type", "Authorization", "X-Signature", "X-Timestamp"])
    expose_headers: list[str] = field(default_factory=list)
    allow_credentials: bool = False
    max_age: int = 86400  # preflight cache 24h

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if *origin* is permitted."""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins

    def headers(self, origin: str = "*") -> dict[str, str]:
        """Return CORS response headers."""
        headers: dict[str, str] = {}
        effective_origin = origin if self.is_origin_allowed(origin) else ""
        if not effective_origin:
            return headers
        headers["Access-Control-Allow-Origin"] = effective_origin
        headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        headers["Access-Control-Max-Age"] = str(self.max_age)
        if self.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        return headers


# ---------------------------------------------------------------------------
# Secure HTTP headers
# ---------------------------------------------------------------------------

class SecureHeaders:
    """Generate security-related HTTP response headers."""

    _DEFAULTS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }

    @classmethod
    def as_dict(cls, overrides: Optional[dict[str, str]] = None) -> dict[str, str]:
        """Return a dict of security headers, with optional overrides."""
        headers = dict(cls._DEFAULTS)
        if overrides:
            headers.update(overrides)
        return headers

    @classmethod
    def apply(cls, response_headers: dict[str, str], overrides: Optional[dict[str, str]] = None) -> None:
        """Merge security headers into an existing headers dict in-place."""
        response_headers.update(cls.as_dict(overrides))


# ---------------------------------------------------------------------------
# Unified SecurityManager facade
# ---------------------------------------------------------------------------

class SecurityManager:
    """Unified security facade that ties together all security primitives."""

    def __init__(
        self,
        signing_secret: str = "",
        cors_config: Optional[CORSConfig] = None,
    ):
        self.audit = AuditLogger()
        self.validator = InputValidator()
        self.xss = XSSSanitiser()
        self.sql_detector = SQLInjectionDetector()
        self.signer = RequestSigner(signing_secret) if signing_secret else None
        self.cors = cors_config or CORSConfig()
        self.headers = SecureHeaders
        logger.info("SecurityManager initialised")

    # -- Convenience audit wrappers -----------------------------------------

    def audit_login(self, actor: str, success: bool, ip: str = "", detail: str = "") -> AuditEntry:
        action = AuditAction.LOGIN_SUCCESS if success else AuditAction.LOGIN_FAILURE
        return self.audit.log(action, actor=actor, source_ip=ip, detail=detail, success=success)

    def audit_permission_denied(self, actor: str, resource: str, ip: str = "") -> AuditEntry:
        return self.audit.log(
            AuditAction.PERMISSION_DENIED,
            actor=actor,
            resource=resource,
            source_ip=ip,
            success=False,
        )

    def audit_rate_limit(self, actor: str, ip: str = "") -> AuditEntry:
        return self.audit.log(AuditAction.RATE_LIMIT_HIT, actor=actor, source_ip=ip, success=False)

    # -- Input validation pipeline ------------------------------------------

    def validate_request(self, data: dict, rules: dict[str, Any]) -> dict:
        """Validate request data against a set of rules.

        *rules* maps field names to dicts with keys:
          type: "str" | "num" | "symbol" | "side"
          required: bool
          max_length: int (for str)
          min_val / max_val: float (for num)
        """
        clean: dict[str, Any] = {}
        for field_name, rule in rules.items():
            value = data.get(field_name)
            if value is None:
                if rule.get("required", False):
                    raise ValueError(f"Missing required field: {field_name}")
                continue

            # SQL injection check
            if isinstance(value, str) and self.sql_detector.is_suspicious(value):
                self.audit.log(
                    AuditAction.SQL_INJECTION_ATTEMPT,
                    resource=field_name,
                    detail=f"Suspicious value blocked for field '{field_name}'",
                    success=False,
                )
                raise ValueError(f"Suspicious input in field: {field_name}")

            # XSS check
            if isinstance(value, str) and self.xss.has_xss(value):
                self.audit.log(
                    AuditAction.XSS_ATTEMPT,
                    resource=field_name,
                    detail=f"XSS-like content blocked in field '{field_name}'",
                    success=False,
                )
                raise ValueError(f"Potentially unsafe input in field: {field_name}")

            ftype = rule.get("type", "str")
            if ftype == "str":
                clean[field_name] = self.validator.validate_string(
                    str(value),
                    max_length=rule.get("max_length", 1024),
                )
            elif ftype == "num":
                clean[field_name] = self.validator.validate_numeric(
                    value,
                    min_val=rule.get("min_val"),
                    max_val=rule.get("max_val"),
                )
            elif ftype == "symbol":
                clean[field_name] = self.validator.validate_symbol(str(value))
            elif ftype == "side":
                clean[field_name] = self.validator.validate_side(str(value))
            else:
                clean[field_name] = value

        return clean

    # -- Signed request verification ----------------------------------------

    def sign_request(self, method: str, path: str, body: bytes = b"") -> dict[str, str]:
        """Sign a request. Returns headers to add."""
        if not self.signer:
            raise RuntimeError("Request signing not configured (no signing_secret)")
        return self.signer.sign(method, path, body)

    def verify_request(
        self,
        method: str,
        path: str,
        body: bytes,
        signature: str,
        timestamp: str,
    ) -> bool:
        """Verify a signed request."""
        if not self.signer:
            raise RuntimeError("Request signing not configured")
        valid = self.signer.verify(method, path, body, signature, timestamp)
        if not valid:
            self.audit.log(
                AuditAction.REQUEST_SIGNATURE_INVALID,
                detail=f"{method} {path}",
                success=False,
            )
        return valid

    # -- Response helpers ---------------------------------------------------

    def safe_response(self, data: dict) -> dict:
        """Sanitise response data to prevent XSS."""
        return self.xss.sanitise_dict(data)

    def cors_headers(self, origin: str = "*") -> dict[str, str]:
        """Return CORS headers for the given origin."""
        return self.cors.headers(origin)

    def security_headers(self) -> dict[str, str]:
        """Return security response headers."""
        return self.headers.as_dict()
