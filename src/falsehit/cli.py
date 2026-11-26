"""Command line: seed, calibrate, replay, serve."""

from __future__ import annotations

from pathlib import Path

import typer

from .calibrate import Calibration, calibrate_directory
from .replay import replay_directory
from .seed import SEED, seed_directory

app = typer.Typer(
    add_completion=False, help="A semantic cache for LLM calls with a bounded false-hit rate."
)


def format_calibration(c: Calibration) -> str:
    h = c.heldout
    lines = [
        f"{'tau':>5} | {'FHR':>7} {'CP bound':>9} {'recall':>7} | "
        f"{'FHR':>7} {'CP bound':>9} {'recall':>7}",
        f"{'':>5} | {'with guards':^25} | {'without guards':^25}",
    ]
    for row in c.table:
        step = round(row.tau * 100)
        if step % 5 and abs(row.tau - c.tau) > 1e-9:
            continue
        mark = "  <- chosen" if abs(row.tau - c.tau) < 1e-9 else ""
        lines.append(
            f"{row.tau:>5.2f} | {row.fhr:>7.4f} {row.upper:>9.4f} {row.recall:>7.3f} | "
            f"{row.fhr_unguarded:>7.4f} {row.upper_unguarded:>9.4f} "
            f"{row.recall_unguarded:>7.3f}{mark}"
        )
    lines += [
        "",
        f"target false-hit rate {c.target_fhr}, one-sided {c.confidence:.0%} Clopper-Pearson bound",
        f"chosen tau {c.tau:.2f}: calibration {c.calibration_accepted}/{c.calibration_near_miss} "
        f"accepted near-misses, bound {c.calibration_upper:.4f}",
        f"held-out at tau {c.tau:.2f}: false-hit rate {h.fhr_guarded:.4f} with guards "
        f"({h.accepted_guarded}/{h.near_miss}), {h.fhr_unguarded:.4f} without "
        f"({h.accepted_unguarded}/{h.near_miss})",
        f"held-out paraphrase recall: {h.recall_guarded:.3f} with guards at tau {c.tau:.2f}",
    ]
    if h.tau_unguarded is not None:
        lines.append(
            f"  without guards tau must be refitted to {h.tau_unguarded:.2f}, "
            f"where recall is {h.recall_unguarded_calibrated:.3f}"
        )
    lines += [
        "",
        "guard ablation on held-out near-misses at the chosen tau:",
        f"{'subset':<10}{'pairs':>6}{'no guards':>11}{'own guard':>11}{'all guards':>12}",
    ]
    for a in c.ablation:
        lines.append(
            f"{a.subset:<10}{a.pairs:>6}{a.accepted_no_guards:>11}{a.accepted_own_guard:>11}"
            f"{a.accepted_all_guards:>12}"
        )
    return "\n".join(lines)


@app.command()
def seed(out: Path = Path("data"), seed_value: int = typer.Option(SEED, "--seed")) -> None:
    """Generate labelled pairs, traffic and tenants into a directory."""
    counts = seed_directory(out, seed_value)
    typer.echo(
        f"wrote {counts['pairs']} pairs and {counts['traffic']} requests "
        f"({counts['intents']} intents) to {out}"
    )


@app.command()
def calibrate(
    target_fhr: float = typer.Option(0.01, "--target-fhr"),
    data: Path = Path("data"),
) -> None:
    """Fit the match threshold tau and write <data>/calibration.json."""
    result = calibrate_directory(data, target_fhr)
    (data / "calibration.json").write_text(result.to_json(), encoding="utf-8")
    typer.echo(format_calibration(result))


@app.command()
def replay(
    traffic: Path,
    tau: float | None = typer.Option(None, help="Override the calibrated threshold."),
    ttl: int = typer.Option(7200, help="Cache entry lifetime in seconds."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Replay recorded traffic through the app in-process and print per-tenant results."""
    report = replay_directory(traffic, tau, ttl)
    typer.echo(
        report.to_json() if json_out else f"tau {report.tau:.2f}, ttl {ttl}s\n" + report.table()
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: Path | None = typer.Option(None, help="Settings YAML (default: $FALSEHIT_CONFIG)."),
) -> None:
    """Run the service with uvicorn."""
    import os

    import uvicorn

    if config is not None:
        os.environ["FALSEHIT_CONFIG"] = str(config)
    uvicorn.run("falsehit.api:create_app_from_env", host=host, port=port, factory=True)
