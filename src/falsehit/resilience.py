"""A circuit breaker around the primary upstream, with a secondary to fall back on."""

from __future__ import annotations

import threading
from enum import StrEnum

from .clock import Clock
from .schemas import ChatRequest
from .upstream import ChatModel, Completion, UpstreamError


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Opens after `threshold` consecutive failures, admits one probe after `reset_after`
    seconds, and closes on that probe's success."""

    def __init__(self, clock: Clock, threshold: int = 3, reset_after: float = 30.0) -> None:
        self._clock = clock
        self._threshold = threshold
        self._reset_after = reset_after
        self._state = State.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    def allow(self) -> bool:
        """True when a call may go to the primary. In half-open state only one caller wins."""
        with self._lock:
            if self._state is State.CLOSED:
                return True
            if (
                self._state is State.OPEN
                and self._clock.now() - self._opened_at >= self._reset_after
            ):
                self._state = State.HALF_OPEN
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._state = State.CLOSED
            self._failures = 0

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is State.HALF_OPEN or self._failures >= self._threshold:
                self._state = State.OPEN
                self._opened_at = self._clock.now()


class FailoverUpstream:
    """Primary behind a breaker; every failed or skipped primary call goes to the secondary."""

    name = "failover"

    def __init__(self, primary: ChatModel, secondary: ChatModel | None, breaker: CircuitBreaker):
        self.primary = primary
        self.secondary = secondary
        self.breaker = breaker
        self.last_served_by = ""

    def complete(self, request: ChatRequest) -> Completion:
        if self.breaker.allow():
            try:
                result = self.primary.complete(request)
            except UpstreamError as exc:
                if not exc.trips_breaker:
                    # The upstream answered, so it is reachable (this also settles a half-open
                    # probe). The caller's request is at fault, so do not fail over.
                    self.breaker.success()
                    raise
                self.breaker.failure()
            else:
                self.breaker.success()
                self.last_served_by = self.primary.name
                return result
        if self.secondary is None:
            raise UpstreamError(503, "primary upstream unavailable and no secondary configured")
        result = self.secondary.complete(request)
        self.last_served_by = self.secondary.name
        return result
