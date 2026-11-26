"""Drive the ASGI app in-process with recorded traffic and score it against true intents.

A hit whose served answer belongs to a different intent than the request's true intent is a
false hit, and the cost it appears to avoid is not booked as a saving.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from ._client import TestClient
from .api import create_app
from .clock import ManualClock
from .config import Settings, TenantConfig
from .intents import ANSWER_TO_INTENT
from .seed import read_jsonl
from .upstream import FakeUpstream

MODEL = "support-large"
EPOCH = 1_760_000_000.0


@dataclass
class TenantReport:
    tenant: str
    requests: int = 0
    hits: int = 0
    misses: int = 0
    bypass: int = 0
    false_hits: int = 0
    cross_tenant_hits: int = 0
    rejected: int = 0
    spend: Decimal = field(default_factory=lambda: Decimal(0))
    gross_avoided: Decimal = field(default_factory=lambda: Decimal(0))
    savings: Decimal = field(default_factory=lambda: Decimal(0))

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def false_hit_rate(self) -> float:
        return self.false_hits / self.hits if self.hits else 0.0


@dataclass
class ReplayReport:
    tau: float
    tenants: dict[str, TenantReport]
    hot_requests: int
    hot_bypassed: int
    outcomes: list[tuple[str, str, str, str]]
    """(request_id, tenant, outcome, served_intent) in traffic order."""
    ledger_records: list = field(default_factory=list, repr=False)

    def total(self) -> TenantReport:
        total = TenantReport("all")
        for t in self.tenants.values():
            for name in (
                "requests",
                "hits",
                "misses",
                "bypass",
                "false_hits",
                "cross_tenant_hits",
                "rejected",
            ):
                setattr(total, name, getattr(total, name) + getattr(t, name))
            total.spend += t.spend
            total.gross_avoided += t.gross_avoided
            total.savings += t.savings
        return total

    def to_json(self) -> str:
        def row(r: TenantReport) -> dict:
            return {
                "requests": r.requests,
                "hits": r.hits,
                "misses": r.misses,
                "bypass": r.bypass,
                "hit_rate": round(r.hit_rate, 4),
                "false_hits": r.false_hits,
                "false_hit_rate": round(r.false_hit_rate, 4),
                "cross_tenant_hits": r.cross_tenant_hits,
                "spend_usd": str(r.spend),
                "savings_usd": str(r.savings),
            }

        data = {
            "tau": self.tau,
            "tenants": {n: row(r) for n, r in self.tenants.items()},
            "total": row(self.total()),
        }
        return json.dumps(data, indent=2) + "\n"

    def table(self) -> str:
        header = (
            f"{'tenant':<12}{'requests':>9}{'hit rate':>10}{'false hits':>11}"
            f"{'spend (USD)':>13}{'savings (USD)':>15}"
        )
        lines = [header, "-" * len(header)]
        for r in [*self.tenants.values(), self.total()]:
            lines.append(
                f"{r.tenant:<12}{r.requests:>9}{r.hit_rate:>10.3f}{r.false_hits:>11}"
                f"{r.spend:>13.4f}{r.savings:>15.4f}"
            )
        return "\n".join(lines)


def load_tenants(data_dir: Path) -> dict[str, dict]:
    return json.loads((Path(data_dir) / "tenants.json").read_text(encoding="utf-8"))


def replay(
    traffic: list[dict],
    tenants: dict[str, dict],
    tau: float,
    ttl_seconds: int = 7200,
) -> ReplayReport:
    settings = Settings(
        tenants={
            name: TenantConfig(
                api_key=spec["api_key"],
                rpm=spec["rpm"],
                monthly_budget_usd=Decimal(spec["monthly_budget_usd"]),
            )
            for name, spec in tenants.items()
        },
        tau=tau,
        ttl_seconds=ttl_seconds,
    )
    clock = ManualClock(EPOCH)
    app = create_app(settings, FakeUpstream("replay"), clock)
    gateway = app.state.gateway

    reports = {name: TenantReport(name) for name in tenants}
    origin: dict[str, str] = {}
    served: dict[str, tuple[str, bool]] = {}
    hot = hot_bypassed = 0
    outcomes: list[tuple[str, str, str, str]] = []

    # One client context keeps a single event-loop portal alive instead of one per request.
    with TestClient(app) as client:
        for rec in traffic:
            clock.set(EPOCH + rec["t"])
            name = rec["tenant"]
            body = {
                "model": MODEL,
                "temperature": rec["temperature"],
                "cache_tag": rec["tag"],
                "messages": [
                    {"role": "system", "content": tenants[name]["system_prompt"]},
                    {"role": "user", "content": rec["text"]},
                ],
            }
            response = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {tenants[name]['api_key']}"},
            )
            report = reports[name]
            report.requests += 1
            if response.status_code != 200:
                report.rejected += 1
                continue
            outcome = response.headers["x-cache"]
            payload = response.json()
            intent = ANSWER_TO_INTENT.get(payload["choices"][0]["message"]["content"], "unknown")
            request_id = response.headers["x-request-id"]
            outcomes.append((request_id, name, outcome, intent))
            if rec["temperature"] > 0.3:
                hot += 1
                hot_bypassed += outcome == "bypass"
            if outcome == "hit":
                report.hits += 1
                if origin.get(payload["id"]) != name:
                    report.cross_tenant_hits += 1
                served[request_id] = (intent, intent == rec["intent"])
                if intent != rec["intent"]:
                    report.false_hits += 1
            else:
                origin[payload["id"]] = name
                report.misses += outcome == "miss"
                report.bypass += outcome == "bypass"

    per_tenant_ok: dict[str, Decimal] = defaultdict(Decimal)
    for record in gateway.ledger.records:
        report = reports[record.tenant]
        report.spend += record.cost_charged
        report.gross_avoided += record.cost_avoided
        if record.outcome == "hit" and served[record.request_id][1]:
            per_tenant_ok[record.tenant] += record.cost_avoided
    for name, report in reports.items():
        report.savings = per_tenant_ok[name]
    return ReplayReport(tau, reports, hot, hot_bypassed, outcomes, list(gateway.ledger.records))


def replay_directory(
    traffic_path: Path, tau: float | None = None, ttl_seconds: int = 7200
) -> ReplayReport:
    traffic_path = Path(traffic_path)
    data_dir = traffic_path.parent
    if tau is None:
        calibration = data_dir / "calibration.json"
        if not calibration.exists():
            raise FileNotFoundError(
                f"{calibration} not found; run `falsehit calibrate` first or pass --tau"
            )
        tau = float(json.loads(calibration.read_text(encoding="utf-8"))["tau"])
    return replay(read_jsonl(traffic_path), load_tenants(data_dir), tau, ttl_seconds)
