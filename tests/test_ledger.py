from decimal import Decimal

from falsehit.ledger import Ledger, LedgerRecord, PriceTable

DAY = 1_760_000_000.0


def rec(i, outcome, charged, avoided, ts=DAY, tenant="alpha"):
    return LedgerRecord(
        f"r{i}", ts, tenant, outcome, "m", 10, 20, Decimal(charged), Decimal(avoided), f"c{i}"
    )


def test_cost_is_exact_decimal_per_million_tokens():
    table = PriceTable.load()
    assert table.cost("support-large", 1_000_000, 1_000_000) == Decimal("18")
    assert table.cost("support-large", 100, 80) == Decimal("0.0015")
    assert "support-small" in table and "nope" not in table


def test_rollups_equal_the_sum_of_records_to_the_cent(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    records = [
        rec(1, "miss", "0.0033", "0"),
        rec(2, "hit", "0", "0.0033"),
        rec(3, "hit", "0", "0.0033", ts=DAY + 90_000),
        rec(4, "bypass", "0.0071", "0"),
        rec(5, "miss", "0.0019", "0", tenant="beta"),
    ]
    for r in records:
        ledger.append(r)
    alpha = [r for r in records if r.tenant == "alpha"]
    total = ledger.totals("alpha")
    assert total.charged == sum(r.cost_charged for r in alpha) == Decimal("0.0104")
    assert total.avoided == sum(r.cost_avoided for r in alpha) == Decimal("0.0066")
    assert (total.requests, total.hits, total.misses, total.bypass) == (4, 2, 1, 1)
    assert len(ledger.daily("alpha")) == 2
    assert ledger.month_spend("alpha", "2025-10") == Decimal("0.0104")
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 5
