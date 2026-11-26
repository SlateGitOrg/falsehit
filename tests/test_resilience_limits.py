from decimal import Decimal

import pytest
from conftest import Harness, make_settings

from falsehit.clock import ManualClock
from falsehit.config import TenantConfig
from falsehit.limits import RateLimiter, over_budget
from falsehit.resilience import CircuitBreaker, FailoverUpstream, State
from falsehit.schemas import ChatRequest
from falsehit.upstream import FakeUpstream, UpstreamError


def test_breaker_opens_after_three_failures_and_probes_once_after_30s(failover_harness):
    h = failover_harness
    questions = [f"how do I export invoices for {y}" for y in (2021, 2022, 2023, 2024, 2025, 2026)]
    for q in questions[:3]:  # primary fails, the secondary answers every one of them
        assert h.ask(q).status_code == 200
    assert (h.primary.calls, h.secondary.calls) == (3, 3)
    assert h.upstream.breaker.state is State.OPEN

    assert h.ask(questions[3]).status_code == 200  # call 4 skips the primary
    assert (h.primary.calls, h.secondary.calls) == (3, 4)

    h.clock.advance(30)
    assert h.ask(questions[4]).status_code == 200  # exactly one half-open probe
    assert (h.primary.calls, h.secondary.calls) == (4, 4)
    assert h.upstream.breaker.state is State.CLOSED
    assert h.ask(questions[5]).status_code == 200
    assert h.primary.calls == 5


def test_half_open_admits_a_single_probe():
    clock = ManualClock()
    breaker = CircuitBreaker(clock, threshold=3, reset_after=30)
    for _ in range(3):
        breaker.failure()
    assert not breaker.allow()
    clock.advance(30)
    assert breaker.allow() and not breaker.allow()  # the second caller is refused
    breaker.failure()  # a failed probe re-opens and restarts the timer
    assert breaker.state is State.OPEN
    clock.advance(29)
    assert not breaker.allow()
    clock.advance(1)
    assert breaker.allow()


def test_a_success_resets_the_failure_count():
    breaker = CircuitBreaker(ManualClock(), threshold=3)
    breaker.failure(), breaker.failure(), breaker.success(), breaker.failure()
    assert breaker.state is State.CLOSED


def test_token_bucket_refills_over_time():
    clock = ManualClock()
    limiter = RateLimiter(clock)
    assert all(limiter.acquire("a", 60) is None for _ in range(60))
    assert limiter.acquire("a", 60) == 1
    assert limiter.acquire("b", 60) is None  # tenants have separate buckets
    clock.advance(1)
    assert limiter.acquire("a", 60) is None


def test_61st_request_in_a_minute_gets_429_with_retry_after(harness):
    codes = [harness.ask("how do I export invoices for 2024").status_code for _ in range(61)]
    assert codes[:60] == [200] * 60
    last = harness.ask("how do I export invoices for 2024")
    assert codes[60] == 429
    assert last.headers["retry-after"] == "1"
    assert last.json()["error"]["code"] == "rate_limit_exceeded"
    assert last.headers["x-cache"] == "bypass"
    harness.clock.advance(1)
    assert harness.ask("how do I export invoices for 2024").status_code == 200


def test_spend_cap_blocks_the_next_miss_but_serves_hits():
    settings = make_settings(
        tenants={
            "alpha": TenantConfig(api_key="key-alpha", rpm=600, monthly_budget_usd=Decimal("0.50"))
        },
        prices={"support-large": {"input": 5000, "output": 5000}},
    )
    h = Harness(settings)
    first = h.ask("how do I export invoices for 2024")
    assert first.status_code == 200 and first.headers["x-cache"] == "miss"
    spent = h.gateway.ledger.month_spend("alpha", "2025-10")
    assert Decimal("0.25") < spent < Decimal("0.50")  # still under the cap
    assert h.ask("where do I reset my password").status_code == 200  # pushes spend over 0.50
    assert h.gateway.ledger.month_spend("alpha", "2025-10") >= Decimal("0.50")
    assert over_budget(h.gateway.ledger.month_spend("alpha", "2025-10"), Decimal("0.50"))

    blocked = h.ask("how do I cancel my subscription")
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "budget_exceeded"
    assert blocked.headers["x-cache"] == "miss"

    hit = h.ask("how do I export invoices for 2024")
    assert hit.status_code == 200 and hit.headers["x-cache"] == "hit"


class _Rejecting:
    """An upstream that answers every call with a 4xx."""

    def __init__(self, status):
        self.name = "rejecting"
        self.status = status
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise UpstreamError(self.status, "rejected")


def _request():
    return ChatRequest(model="support-large", messages=[{"role": "user", "content": "hi"}])


def test_trip_rules_by_status():
    assert [UpstreamError(s, "").trips_breaker for s in (500, 502, 503, 429)] == [True] * 4
    assert [UpstreamError(s, "").trips_breaker for s in (400, 401, 404, 422)] == [False] * 4


def test_client_errors_never_open_the_breaker_or_fail_over():
    clock = ManualClock()
    primary, secondary = _Rejecting(400), FakeUpstream("secondary")
    upstream = FailoverUpstream(primary, secondary, CircuitBreaker(clock))
    for _ in range(5):
        with pytest.raises(UpstreamError) as exc:
            upstream.complete(_request())
        assert exc.value.status == 400
    assert upstream.breaker.state is State.CLOSED
    assert (primary.calls, secondary.calls) == (5, 0)


def test_client_errors_do_not_count_toward_consecutive_failures():
    clock = ManualClock()
    primary = _Rejecting(404)
    upstream = FailoverUpstream(primary, FakeUpstream("secondary"), CircuitBreaker(clock))
    upstream.breaker.failure()
    upstream.breaker.failure()
    with pytest.raises(UpstreamError):
        upstream.complete(_request())
    upstream.breaker.failure()  # would be the third consecutive failure if the 404 had not reset
    assert upstream.breaker.state is State.CLOSED


def test_a_4xx_on_the_half_open_probe_settles_the_breaker():
    clock = ManualClock()
    breaker = CircuitBreaker(clock)
    for _ in range(3):
        breaker.failure()
    clock.advance(30)
    upstream = FailoverUpstream(_Rejecting(400), FakeUpstream("secondary"), breaker)
    with pytest.raises(UpstreamError):
        upstream.complete(_request())
    assert breaker.state is State.CLOSED


def test_429_and_5xx_from_a_real_client_open_the_breaker():
    clock = ManualClock()
    primary = _Rejecting(429)
    secondary = FakeUpstream("secondary")
    upstream = FailoverUpstream(primary, secondary, CircuitBreaker(clock))
    for _ in range(3):
        upstream.complete(_request())
    assert upstream.breaker.state is State.OPEN and secondary.calls == 3
