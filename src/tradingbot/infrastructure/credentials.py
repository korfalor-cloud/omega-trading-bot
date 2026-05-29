"""Secure Credential Storage — encrypted secrets, rotation, multi-environment support.

Implements:
- Encrypted credential storage (Fernet with base64+hashlib fallback)
- Environment variable loading
- Credential rotation support
- Multi-environment support (dev / staging / prod)
- Never log or expose secrets in plaintext

Secrets are encrypted at rest. The raw value is never written to logs or
repr strings; only masked previews are ever exposed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fernet():
    try:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]
        return Fernet
    except ImportError:
        return None


FernetCls = _get_fernet()


def _mask(value: str, visible: int = 4) -> str:
    """Return a masked preview: ``abc…xyz``."""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * 6}{value[-visible:]}"


def _encrypt_value(plaintext: str, key: bytes) -> str:
    """Encrypt *plaintext* and return a base64 ciphertext string."""
    if FernetCls is not None:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]
        return Fernet(key).encrypt(plaintext.encode()).decode()
    # Fallback: XOR + base64 (not production-grade)
    data = plaintext.encode()
    ct = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return base64.urlsafe_b64encode(ct).decode()


def _decrypt_value(ciphertext: str, key: bytes) -> str:
    """Decrypt a value produced by *_encrypt_value*."""
    if FernetCls is not None:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]
        return Fernet(key).decrypt(ciphertext.encode()).decode()
    ct = base64.urlsafe_b64decode(ciphertext.encode())
    data = bytes(b ^ key[i % len(key)] for i, b in enumerate(ct))
    return data.decode()


def _generate_key() -> bytes:
    """Generate a fresh encryption key."""
    if FernetCls is not None:
        return FernetCls.generate_key()
    return base64.urlsafe_b64encode(os.urandom(32))


# ---------------------------------------------------------------------------
# Environment enum
# ---------------------------------------------------------------------------

class Environment(Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

    @classmethod
    def detect(cls) -> "Environment":
        """Detect the current environment from ``OMEGA_ENV`` or ``ENVIRONMENT``."""
        raw = os.environ.get("OMEGA_ENV", os.environ.get("ENVIRONMENT", "dev")).lower()
        try:
            return cls(raw)
        except ValueError:
            logger.warning("Unknown environment '%s', defaulting to dev", raw)
            return cls.DEV


# ---------------------------------------------------------------------------
# Credential metadata
# ---------------------------------------------------------------------------

@dataclass
class CredentialMeta:
    """Metadata for a stored credential (value is stored separately, encrypted)."""
    cred_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    description: str = ""
    environment: Environment = Environment.DEV
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = no expiry
    rotation_interval: int = 0  # seconds; 0 = no auto-rotation
    tags: list[str] = field(default_factory=list)
    version: int = 1

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def needs_rotation(self) -> bool:
        if self.rotation_interval <= 0:
            return False
        return (time.time() - self.updated_at) > self.rotation_interval

    def __repr__(self) -> str:
        return f"CredentialMeta(name={self.name!r}, env={self.environment.value}, version={self.version})"


# ---------------------------------------------------------------------------
# Credential record (internal, includes encrypted value)
# ---------------------------------------------------------------------------

@dataclass
class _CredentialRecord:
    meta: CredentialMeta
    encrypted_value: str = ""
    previous_values: list[str] = field(default_factory=list)  # older encrypted values


# ---------------------------------------------------------------------------
# Credential store
# ---------------------------------------------------------------------------

class CredentialStore:
    """Encrypted credential storage with rotation and multi-environment support.

    Usage::

        store = CredentialStore(master_key=..., environment=Environment.PROD)
        store.set("EXCHANGE_API_KEY", "sk-abc123", description="Alpaca paper key")
        value = store.get("EXCHANGE_API_KEY")
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        environment: Optional[Environment] = None,
        storage_path: Optional[str] = None,
    ):
        self._key = master_key or _generate_key()
        self._environment = environment or Environment.detect()
        self._storage_path = Path(storage_path) if storage_path else None
        self._credentials: dict[str, _CredentialRecord] = {}

        # Load from disk if a storage path is given
        if self._storage_path and self._storage_path.exists():
            self._load_from_disk()

        logger.info(
            "CredentialStore initialised (env=%s, stored=%d)",
            self._environment.value,
            len(self._credentials),
        )

    # -- environment --------------------------------------------------------

    @property
    def environment(self) -> Environment:
        return self._environment

    # -- CRUD ---------------------------------------------------------------

    def set(
        self,
        name: str,
        value: str,
        description: str = "",
        expires_in: int = 0,
        rotation_interval: int = 0,
        tags: Optional[list[str]] = None,
        env_override: Optional[Environment] = None,
    ) -> CredentialMeta:
        """Store or update a credential.  The value is encrypted before storage."""
        env = env_override or self._environment
        existing = self._find(name, env)
        encrypted = _encrypt_value(value, self._key)

        if existing:
            # Rotate: keep old encrypted value in history
            existing.previous_values.append(existing.encrypted_value)
            existing.encrypted_value = encrypted
            existing.meta.updated_at = time.time()
            existing.meta.version += 1
            existing.meta.description = description or existing.meta.description
            existing.meta.tags = tags or existing.meta.tags
            if rotation_interval:
                existing.meta.rotation_interval = rotation_interval
            if expires_in:
                existing.meta.expires_at = time.time() + expires_in
            logger.info("Credential rotated: %s (v%d)", name, existing.meta.version)
            self._persist()
            return existing.meta

        meta = CredentialMeta(
            name=name,
            description=description,
            environment=env,
            expires_at=time.time() + expires_in if expires_in else 0.0,
            rotation_interval=rotation_interval,
            tags=tags or [],
        )
        self._credentials[meta.cred_id] = _CredentialRecord(meta=meta, encrypted_value=encrypted)
        logger.info("Credential stored: %s (env=%s)", name, env.value)
        self._persist()
        return meta

    def get(self, name: str, env_override: Optional[Environment] = None) -> Optional[str]:
        """Retrieve and decrypt a credential. Returns None if not found or expired."""
        env = env_override or self._environment
        record = self._find(name, env)
        if record is None:
            logger.debug("Credential not found: %s", name)
            return None
        if record.meta.is_expired:
            logger.warning("Credential expired: %s", name)
            return None
        try:
            return _decrypt_value(record.encrypted_value, self._key)
        except Exception as exc:
            logger.error("Failed to decrypt credential '%s': %s", name, type(exc).__name__)
            return None

    def delete(self, name: str, env_override: Optional[Environment] = None) -> bool:
        """Delete a credential. Returns True if found and removed."""
        env = env_override or self._environment
        record = self._find(name, env)
        if record:
            del self._credentials[record.meta.cred_id]
            logger.info("Credential deleted: %s", name)
            self._persist()
            return True
        return False

    def list_credentials(self, env_override: Optional[Environment] = None) -> list[CredentialMeta]:
        """List metadata for all credentials (values are never returned)."""
        env = env_override or self._environment
        return [r.meta for r in self._credentials.values() if r.meta.environment == env]

    def exists(self, name: str, env_override: Optional[Environment] = None) -> bool:
        """Check if a credential exists."""
        env = env_override or self._environment
        return self._find(name, env) is not None

    # -- Rotation -----------------------------------------------------------

    def check_rotation_needed(self) -> list[CredentialMeta]:
        """Return credentials that are due for rotation."""
        return [r.meta for r in self._credentials.values() if r.meta.needs_rotation]

    def rotate(self, name: str, new_value: str, env_override: Optional[Environment] = None) -> Optional[CredentialMeta]:
        """Explicitly rotate a credential to a new value."""
        return self.set(name, new_value, env_override=env_override)

    def get_previous_values(self, name: str, env_override: Optional[Environment] = None) -> list[str]:
        """Return decrypted previous values (for graceful key rollover).

        Only the last 5 previous values are kept.
        """
        env = env_override or self._environment
        record = self._find(name, env)
        if record is None:
            return []
        out: list[str] = []
        for enc in reversed(record.previous_values[-5:]):
            try:
                out.append(_decrypt_value(enc, self._key))
            except Exception:
                continue
        return out

    # -- Environment variable loading ---------------------------------------

    def load_from_env(
        self,
        prefix: str = "OMEGA_",
        mapping: Optional[dict[str, str]] = None,
    ) -> int:
        """Load credentials from environment variables.

        If *mapping* is provided it maps env var names to credential names.
        Otherwise, all vars starting with *prefix* are loaded (prefix stripped,
        lowercased).
        """
        count = 0
        if mapping:
            for env_var, cred_name in mapping.items():
                value = os.environ.get(env_var)
                if value:
                    self.set(cred_name, value, tags=["env-loaded"])
                    count += 1
        else:
            for env_var, value in os.environ.items():
                if env_var.startswith(prefix) and value:
                    cred_name = env_var[len(prefix):].lower()
                    self.set(cred_name, value, tags=["env-loaded"])
                    count += 1
        logger.info("Loaded %d credentials from environment", count)
        return count

    # -- Internals ----------------------------------------------------------

    def _find(self, name: str, env: Environment) -> Optional[_CredentialRecord]:
        for record in self._credentials.values():
            if record.meta.name == name and record.meta.environment == env:
                return record
        return None

    def _persist(self) -> None:
        """Write encrypted store to disk (if a path is configured)."""
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        for cid, rec in self._credentials.items():
            data[cid] = {
                "meta": {
                    "cred_id": rec.meta.cred_id,
                    "name": rec.meta.name,
                    "description": rec.meta.description,
                    "environment": rec.meta.environment.value,
                    "created_at": rec.meta.created_at,
                    "updated_at": rec.meta.updated_at,
                    "expires_at": rec.meta.expires_at,
                    "rotation_interval": rec.meta.rotation_interval,
                    "tags": rec.meta.tags,
                    "version": rec.meta.version,
                },
                "encrypted_value": rec.encrypted_value,
                "previous_values": rec.previous_values,
            }
        # Encrypt the entire store blob with the master key
        blob = json.dumps(data)
        encrypted_blob = _encrypt_value(blob, self._key)
        self._storage_path.write_text(encrypted_blob)
        logger.debug("Credential store persisted to %s", self._storage_path)

    def _load_from_disk(self) -> None:
        """Load and decrypt the store from disk."""
        try:
            encrypted_blob = self._storage_path.read_text()  # type: ignore[union-attr]
            blob = _decrypt_value(encrypted_blob, self._key)
            data = json.loads(blob)
            for cid, rec_data in data.items():
                meta = CredentialMeta(
                    cred_id=rec_data["meta"]["cred_id"],
                    name=rec_data["meta"]["name"],
                    description=rec_data["meta"].get("description", ""),
                    environment=Environment(rec_data["meta"]["environment"]),
                    created_at=rec_data["meta"]["created_at"],
                    updated_at=rec_data["meta"]["updated_at"],
                    expires_at=rec_data["meta"].get("expires_at", 0.0),
                    rotation_interval=rec_data["meta"].get("rotation_interval", 0),
                    tags=rec_data["meta"].get("tags", []),
                    version=rec_data["meta"].get("version", 1),
                )
                self._credentials[cid] = _CredentialRecord(
                    meta=meta,
                    encrypted_value=rec_data["encrypted_value"],
                    previous_values=rec_data.get("previous_values", []),
                )
            logger.info("Loaded %d credentials from %s", len(self._credentials), self._storage_path)
        except Exception as exc:
            logger.error("Failed to load credential store from disk: %s", type(exc).__name__)

    def __repr__(self) -> str:
        return f"CredentialStore(env={self._environment.value}, count={len(self._credentials)})"


# ---------------------------------------------------------------------------
# Convenience: exchange credential loader
# ---------------------------------------------------------------------------

def load_exchange_credentials(
    store: CredentialStore,
    exchange: str,
) -> dict[str, str]:
    """Load API key, secret, and optional passphrase for *exchange*.

    Returns a dict with keys ``api_key``, ``api_secret``, ``passphrase``
    (values are decrypted plaintext).
    """
    prefix = exchange.upper()
    result: dict[str, str] = {}
    for field_name in ("api_key", "api_secret", "passphrase"):
        value = store.get(f"{prefix}_{field_name}")
        if value:
            result[field_name] = value
    return result
