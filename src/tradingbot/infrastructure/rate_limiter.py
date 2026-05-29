"""Rate Limiter — exchange API rate limiting.

Implements:
- Token bucket algorithm
- Per-endpoint rate limits
- Automatic backoff on 429 errors
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int = 100
    window: float = 60.0  # seconds


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_time = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """Time until tokens are available."""
        if self.tokens >= tokens:
            return 0
        return (tokens - self.tokens) / self.rate


class RateLimiter:
    """Multi-endpoint rate limiter."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.default_limit = RateLimit(
            requests=config.get("default_requests", 100),
            window=config.get("default_window", 60),
        )
        self._buckets: dict[str, TokenBucket] = {}
        self._limits: dict[str, RateLimit] = {}
        self._backoff_until: dict[str, float] = {}

    def set_limit(self, endpoint: str, requests: int, window: float = 60) -> None:
        self._limits[endpoint] = RateLimit(requests=requests, window=window)
        rate = requests / window
        self._buckets[endpoint] = TokenBucket(rate, requests)

    def acquire(self, endpoint: str = "default", tokens: int = 1) -> bool:
        """Try to acquire permission for a request."""
        # Check backoff
        if endpoint in self._backoff_until:
            if time.time() < self._backoff_until[endpoint]:
                return False
            del self._backoff_until[endpoint]

        # Get or create bucket
        if endpoint not in self._buckets:
            limit = self._limits.get(endpoint, self.default_limit)
            rate = limit.requests / limit.window
            self._buckets[endpoint] = TokenBucket(rate, limit.requests)

        return self._buckets[endpoint].acquire(tokens)

    def wait(self, endpoint: str = "default", tokens: int = 1) -> float:
        """Get wait time until tokens available."""
        if endpoint not in self._buckets:
            return 0
        return self._buckets[endpoint].wait_time(tokens)

    def backoff(self, endpoint: str, seconds: float) -> None:
        """Apply backoff after rate limit hit."""
        self._backoff_until[endpoint] = time.time() + seconds
        logger.warning(f"Rate limit backoff: {endpoint} for {seconds}s")

    def is_available(self, endpoint: str = "default") -> bool:
        """Check if endpoint is available (not in backoff)."""
        if endpoint in self._backoff_until:
            return time.time() >= self._backoff_until[endpoint]
        return True
