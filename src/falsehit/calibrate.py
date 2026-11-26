"""Fit the match threshold from labelled pairs with a Clopper-Pearson bound on the false-hit rate.

False-hit rate (FHR) is accepted near-miss pairs divided by near-miss pairs, where a pair is
accepted when its cosine reaches tau and, with guards on, no guard vetoes it. The chosen tau is
the smallest candidate whose one-sided 95% Clopper-Pearson upper bound on the calibration-split
FHR is at or below the target. The held-out split is then scored at that tau, with and without
guards, and the guards are ablated per near-miss subset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .embed import HashingEmbedder, cosine
from .guards import ALL_GUARDS, Vocabulary, all_vetoes, signature
from .seed import read_jsonl

CONFIDENCE = 0.95
SUBSETS = ("number", "negation", "object")
GUARD_FOR_SUBSET = {
    "number": "number_mismatch",
    "negation": "negation_mismatch",
    "object": "object_mismatch",
}


def clopper_pearson_upper(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided upper confidence bound on a binomial proportion, k successes in n trials."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


@dataclass(frozen=True)
class ScoredPair:
    label: str
    subset: str | None
    cosine: float
    vetoes: tuple[str, ...]

    def accepted(self, tau: float, guards: bool | str = True) -> bool:
        """`guards` may be True (all), False (none) or the name of a single guard."""
        if self.cosine < tau - 1e-9:
            return False
        if guards is False:
            return True
        if guards is True:
            return not self.vetoes
        return guards not in self.vetoes


def score_pairs(
    rows: list[dict], embedder: HashingEmbedder, vocabulary: Vocabulary
) -> dict[str, list[ScoredPair]]:
    """Cosine and guard verdicts for every labelled pair, grouped by split."""
    out: dict[str, list[ScoredPair]] = {"calibration": [], "heldout": []}
    for row in rows:
        sim = cosine(embedder.embed(row["a"]), embedder.embed(row["b"]))
        vetoes = all_vetoes(signature(row["a"], vocabulary), signature(row["b"], vocabulary))
        out[row["split"]].append(ScoredPair(row["label"], row["subset"], sim, vetoes))
    return out


def _rates(
    pairs: list[ScoredPair], tau: float, guards: bool | str = True
) -> tuple[int, int, int, int]:
    near = [p for p in pairs if p.label == "near_miss"]
    para = [p for p in pairs if p.label == "paraphrase"]
    return (
        sum(p.accepted(tau, guards) for p in near),
        len(near),
        sum(p.accepted(tau, guards) for p in para),
        len(para),
    )


@dataclass
class Candidate:
    tau: float
    accepted: int
    fhr: float
    upper: float
    recall: float
    accepted_unguarded: int
    fhr_unguarded: float
    upper_unguarded: float
    recall_unguarded: float


@dataclass
class HeldOut:
    near_miss: int
    paraphrase: int
    accepted_guarded: int
    fhr_guarded: float
    accepted_unguarded: int
    fhr_unguarded: float
    recall_guarded: float
    recall_unguarded_at_tau: float
    tau_unguarded: float | None
    recall_unguarded_calibrated: float | None
    """Recall if no guards are deployed and tau is refitted without them."""


@dataclass
class SubsetAblation:
    subset: str
    pairs: int
    accepted_no_guards: int
    accepted_own_guard: int
    accepted_all_guards: int
    fhr_no_guards: float
    fhr_all_guards: float


@dataclass
class Calibration:
    target_fhr: float
    confidence: float
    tau: float
    calibration_accepted: int
    calibration_near_miss: int
    calibration_fhr: float
    calibration_upper: float
    table: list[Candidate]
    heldout: HeldOut
    ablation: list[SubsetAblation] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def default_grid(low: float = 0.30, high: float = 1.00, step: float = 0.01) -> list[float]:
    return [round(float(t), 4) for t in np.arange(low, high + step / 2, step)]


def _fit_tau(
    pairs: list[ScoredPair], grid: list[float], target: float, guards: bool
) -> float | None:
    for tau in grid:
        k, n, _, _ = _rates(pairs, tau, guards)
        if clopper_pearson_upper(k, n) <= target:
            return tau
    return None


def calibrate(
    scored: dict[str, list[ScoredPair]],
    target_fhr: float = 0.01,
    grid: list[float] | None = None,
) -> Calibration:
    grid = grid or default_grid()
    cal, held = scored["calibration"], scored["heldout"]
    tau = _fit_tau(cal, grid, target_fhr, guards=True)
    if tau is None:
        raise ValueError(f"no threshold in the grid reaches a false-hit bound of {target_fhr}")
    tau_unguarded = _fit_tau(cal, grid, target_fhr, guards=False)

    table = []
    for t in grid:
        k, n, hit, para = _rates(cal, t, True)
        ku, _, hitu, _ = _rates(cal, t, False)
        table.append(
            Candidate(
                t,
                k,
                k / n,
                clopper_pearson_upper(k, n),
                hit / para,
                ku,
                ku / n,
                clopper_pearson_upper(ku, n),
                hitu / para,
            )
        )

    k, n, hit, para = _rates(cal, tau, True)
    hk, hn, hhit, hpara = _rates(held, tau, True)
    hku, _, hhitu, _ = _rates(held, tau, False)
    recall_u_cal = None
    if tau_unguarded is not None:
        _, _, r, p = _rates(held, tau_unguarded, False)
        recall_u_cal = r / p
    heldout = HeldOut(
        near_miss=hn,
        paraphrase=hpara,
        accepted_guarded=hk,
        fhr_guarded=hk / hn,
        accepted_unguarded=hku,
        fhr_unguarded=hku / hn,
        recall_guarded=hhit / hpara,
        recall_unguarded_at_tau=hhitu / hpara,
        tau_unguarded=tau_unguarded,
        recall_unguarded_calibrated=recall_u_cal,
    )

    ablation = []
    for subset in SUBSETS:
        group = [p for p in held if p.subset == subset]
        if not group:
            continue
        none = sum(p.accepted(tau, False) for p in group)
        own = sum(p.accepted(tau, GUARD_FOR_SUBSET[subset]) for p in group)
        every = sum(p.accepted(tau, True) for p in group)
        ablation.append(
            SubsetAblation(
                subset, len(group), none, own, every, none / len(group), every / len(group)
            )
        )
    return Calibration(
        target_fhr,
        CONFIDENCE,
        tau,
        k,
        n,
        k / n,
        clopper_pearson_upper(k, n),
        table,
        heldout,
        ablation,
    )


def calibrate_directory(
    data_dir: Path, target_fhr: float = 0.01, vocabulary: Vocabulary | None = None
) -> Calibration:
    rows = read_jsonl(Path(data_dir) / "pairs.jsonl")
    scored = score_pairs(rows, HashingEmbedder(), vocabulary or Vocabulary())
    return calibrate(scored, target_fhr)


assert set(ALL_GUARDS) == set(GUARD_FOR_SUBSET.values())
