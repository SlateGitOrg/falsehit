"""Per-tenant request rate limiting (token bucket) and monthly spend caps."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from decimal import Decimal

from .clock import Clock


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """A token bucket per tenant: capacity `rpm`, refilled continuously at rpm/60 per second."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, tenant: str, rpm: int) -> int | None:
        """Take one token. Returns None on success, else whole seconds until one is free."""
        now = self._clock.now()
        rate = rpm / 60.0
        with self._lock:
            bucket = self._buckets.setdefault(tenant, _Bucket(float(rpm), now))
            bucket.tokens = min(float(rpm), bucket.tokens + (now - bucket.updated) * rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return None
            return max(1, math.ceil((1.0 - bucket.tokens) / rate))


def over_budget(spent: Decimal, cap: Decimal) -> bool:
    """A miss is blocked once this month's charged spend has reached the cap."""
    return spent >= cap
