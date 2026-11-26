"""Service settings: tenants and their API keys, thresholds, limits and prices."""

from __future__ import annotations

import hmac
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TenantConfig(BaseModel):
    api_key: str
    rpm: int = Field(default=60, gt=0)
    monthly_budget_usd: Decimal = Field(default=Decimal("100"), ge=0)


class Settings(BaseModel):
    tenants: dict[str, TenantConfig]
    tau: float = Field(default=0.8, ge=0, le=1)
    """Cosine threshold. `falsehit calibrate` fits it from labelled near-miss pairs."""
    ttl_seconds: int = Field(default=7200, gt=0)
    max_entries_per_partition: int = Field(default=4096, gt=0)
    breaker_failures: int = Field(default=3, gt=0)
    breaker_reset_seconds: float = Field(default=30.0, gt=0)
    prices_path: Path | None = None
    prices: dict[str, dict[str, Any]] | None = None
    """Inline price table; takes precedence over `prices_path`."""
    vocabulary: dict[str, dict[str, list[str]]] | None = None
    ledger_path: Path | None = None

    def tenant_for_key(self, api_key: str) -> str | None:
        """Constant-time lookup of the tenant that owns a bearer key."""
        found = None
        for name, tenant in self.tenants.items():
            if hmac.compare_digest(tenant.api_key.encode(), api_key.encode()):
                found = name
        return found


def load_settings(path: str | Path | None = None) -> Settings:
    """Read settings from a YAML file (default: $FALSEHIT_CONFIG)."""
    path = path or os.environ.get("FALSEHIT_CONFIG")
    if not path:
        raise RuntimeError("no config: pass --config or set FALSEHIT_CONFIG")
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def tau_from_calibration(path: str | Path) -> float:
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["tau"])
