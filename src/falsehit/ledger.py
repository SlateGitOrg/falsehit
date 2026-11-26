"""An append-only cost ledger in Decimal, with per-tenant daily rollups."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml

from .clock import day_of, month_of

Outcome = Literal["hit", "miss", "bypass"]
MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    input: Decimal
    output: Decimal
    """USD per one million tokens."""


class PriceTable:
    def __init__(self, prices: dict[str, ModelPrice]) -> None:
        self._prices = prices

    @classmethod
    def from_mapping(cls, data: dict[str, dict[str, object]]) -> PriceTable:
        return cls(
            {
                m: ModelPrice(Decimal(str(p["input"])), Decimal(str(p["output"])))
                for m, p in data.items()
            }
        )

    @classmethod
    def load(cls, path: Path | None = None) -> PriceTable:
        if path is None:
            text = resources.files("falsehit").joinpath("prices.yaml").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        return cls.from_mapping(yaml.safe_load(text)["models"])

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        price = self._prices[model]
        return (
            Decimal(prompt_tokens) * price.input + Decimal(completion_tokens) * price.output
        ) / MILLION


@dataclass(frozen=True)
class LedgerRecord:
    request_id: str
    ts: float
    tenant: str
    outcome: Outcome
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_charged: Decimal
    cost_avoided: Decimal
    response_id: str

    def to_json(self) -> str:
        data = asdict(self)
        data["cost_charged"] = str(self.cost_charged)
        data["cost_avoided"] = str(self.cost_avoided)
        return json.dumps(data, sort_keys=True)


@dataclass
class Rollup:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    bypass: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    charged: Decimal = field(default_factory=lambda: Decimal(0))
    avoided: Decimal = field(default_factory=lambda: Decimal(0))

    def add(self, record: LedgerRecord) -> None:
        self.requests += 1
        if record.outcome == "hit":
            self.hits += 1
        elif record.outcome == "miss":
            self.misses += 1
        else:
            self.bypass += 1
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.charged += record.cost_charged
        self.avoided += record.cost_avoided


class Ledger:
    """Records are only ever appended. Rollups are derived incrementally from the same records."""

    def __init__(self, path: Path | None = None) -> None:
        self.records: list[LedgerRecord] = []
        self._daily: dict[tuple[str, str], Rollup] = defaultdict(Rollup)
        self._monthly_spend: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
        self._path = path
        self._lock = threading.Lock()

    def append(self, record: LedgerRecord) -> None:
        with self._lock:
            self.records.append(record)
            self._daily[(record.tenant, day_of(record.ts))].add(record)
            self._monthly_spend[(record.tenant, month_of(record.ts))] += record.cost_charged
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(record.to_json() + "\n")

    def month_spend(self, tenant: str, month: str) -> Decimal:
        with self._lock:
            return self._monthly_spend[(tenant, month)]

    def daily(self, tenant: str) -> dict[str, Rollup]:
        with self._lock:
            return {day: r for (t, day), r in sorted(self._daily.items()) if t == tenant}

    def totals(self, tenant: str) -> Rollup:
        total = Rollup()
        for rollup in self.daily(tenant).values():
            total.requests += rollup.requests
            total.hits += rollup.hits
            total.misses += rollup.misses
            total.bypass += rollup.bypass
            total.prompt_tokens += rollup.prompt_tokens
            total.completion_tokens += rollup.completion_tokens
            total.charged += rollup.charged
            total.avoided += rollup.avoided
        return total
