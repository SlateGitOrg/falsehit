"""Injectable clocks so TTL, breaker timeouts and token buckets are testable."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds since the epoch."""


class SystemClock:
    def now(self) -> float:
        return time.time()


class ManualClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 1_760_000_000.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def set(self, value: float) -> None:
        self._now = float(value)

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


def month_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m")


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
