from __future__ import annotations

from decimal import Decimal

import pytest

from falsehit._client import TestClient
from falsehit.api import create_app
from falsehit.calibrate import calibrate_directory
from falsehit.clock import ManualClock
from falsehit.config import Settings, TenantConfig
from falsehit.replay import replay_directory
from falsehit.resilience import CircuitBreaker, FailoverUpstream
from falsehit.seed import seed_directory
from falsehit.upstream import FakeUpstream

SYSTEM = "You are the support assistant."


@pytest.fixture(scope="session")
def seeded(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    seed_directory(out)
    return out


@pytest.fixture(scope="session")
def calibration(seeded):
    return calibrate_directory(seeded, 0.01)


@pytest.fixture(scope="session")
def replay_report(seeded, calibration):
    return replay_directory(seeded / "traffic.jsonl", calibration.tau)


def make_settings(**overrides) -> Settings:
    base = {
        "tenants": {
            "alpha": TenantConfig(api_key="key-alpha", rpm=60, monthly_budget_usd=Decimal("100")),
            "beta": TenantConfig(api_key="key-beta", rpm=60, monthly_budget_usd=Decimal("100")),
        },
        "tau": 0.54,
        "ttl_seconds": 600,
    }
    base.update(overrides)
    return Settings(**base)


class Harness:
    def __init__(self, settings: Settings | None = None, upstream=None):
        self.clock = ManualClock()
        self.upstream = upstream or FakeUpstream("primary")
        self.app = create_app(settings or make_settings(), self.upstream, self.clock)
        self.client = TestClient(self.app)
        self.gateway = self.app.state.gateway

    def ask(self, text, key="key-alpha", system=SYSTEM, **extra):
        body = {
            "model": "support-large",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            **extra,
        }
        return self.client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
        )


@pytest.fixture
def harness():
    return Harness()


@pytest.fixture
def failover_harness():
    """Primary fails on calls 1-3; the secondary always works."""
    clock = ManualClock()
    primary = FakeUpstream("primary", fail_on={1, 2, 3})
    secondary = FakeUpstream("secondary")
    upstream = FailoverUpstream(primary, secondary, CircuitBreaker(clock))
    h = Harness(upstream=upstream)
    h.clock = clock
    h.app = create_app(make_settings(), upstream, clock)
    h.client = TestClient(h.app)
    h.gateway = h.app.state.gateway
    h.primary, h.secondary = primary, secondary
    return h
