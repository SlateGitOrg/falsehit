import json
from decimal import Decimal

from typer.testing import CliRunner

from falsehit.cli import app
from falsehit.replay import replay_directory


def test_replay_has_no_cross_tenant_hits(replay_report):
    assert replay_report.total().cross_tenant_hits == 0
    assert replay_report.total().hits > 1000


def test_every_hot_request_is_a_bypass(replay_report):
    assert replay_report.hot_requests == 500
    assert replay_report.hot_bypassed == 500


def test_replay_hit_rate_and_false_hit_rate(replay_report):
    total = replay_report.total()
    assert total.requests == 5000 and total.rejected == 0
    assert 0.35 <= total.hit_rate <= 0.65
    assert total.false_hit_rate <= 0.02
    assert total.hits + total.misses + total.bypass == 5000


def test_ledger_totals_equal_per_request_records_to_the_cent(replay_report):
    records = replay_report.ledger_records
    assert len(records) == 5000
    for tenant, report in replay_report.tenants.items():
        mine = [r for r in records if r.tenant == tenant]
        assert report.spend == sum(r.cost_charged for r in mine)
        assert report.gross_avoided == sum(r.cost_avoided for r in mine)
        assert report.spend.quantize(Decimal("0.01")) == sum(
            (r.cost_charged for r in mine), Decimal(0)
        ).quantize(Decimal("0.01"))


def test_false_hits_are_excluded_from_savings(replay_report):
    total = replay_report.total()
    assert total.savings <= total.gross_avoided
    if total.false_hits:
        assert total.savings < total.gross_avoided
    else:
        assert total.savings == total.gross_avoided


def test_savings_are_booked_only_for_correct_hits(seeded, calibration):
    """Forcing a very low threshold creates false hits, and the savings drop accordingly."""
    loose = replay_directory(seeded / "traffic.jsonl", tau=0.05)
    assert loose.total().false_hits > 50
    assert loose.total().savings < loose.total().gross_avoided


def test_cli_seed_calibrate_replay(tmp_path):
    runner = CliRunner()
    out = tmp_path / "d"
    assert runner.invoke(app, ["seed", "--out", str(out)]).exit_code == 0
    cal = runner.invoke(app, ["calibrate", "--target-fhr", "0.01", "--data", str(out)])
    assert cal.exit_code == 0 and "chosen tau" in cal.output
    tau = json.loads((out / "calibration.json").read_text())["tau"]
    rep = runner.invoke(app, ["replay", str(out / "traffic.jsonl"), "--json"])
    assert rep.exit_code == 0
    assert json.loads(rep.output)["tau"] == tau
    assert json.loads(rep.output)["total"]["requests"] == 5000
