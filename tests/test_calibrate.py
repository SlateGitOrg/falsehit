import pytest

from falsehit.calibrate import (
    ScoredPair,
    calibrate,
    clopper_pearson_upper,
    default_grid,
)


def test_clopper_pearson_known_values():
    assert round(clopper_pearson_upper(3, 500), 5) == 0.01543
    assert round(clopper_pearson_upper(0, 300), 5) == 0.00994
    assert clopper_pearson_upper(5, 5) == 1.0
    assert clopper_pearson_upper(0, 0) == 1.0


def test_clopper_pearson_is_monotone_in_k():
    bounds = [clopper_pearson_upper(k, 400) for k in range(10)]
    assert bounds == sorted(bounds)


def _fixture_pairs():
    """100 near-misses at cosines 0.50..0.99 and 100 paraphrases at 0.30..0.99."""
    near = [ScoredPair("near_miss", "object", 0.50 + 0.005 * i, ()) for i in range(100)]
    para = [ScoredPair("paraphrase", None, 0.30 + 0.007 * i, ()) for i in range(100)]
    return near + para


def test_fit_picks_smallest_tau_meeting_the_bound():
    pairs = _fixture_pairs()
    result = calibrate({"calibration": pairs, "heldout": pairs}, target_fhr=0.05)
    # With n=100 the 95% bound is <= 0.05 only for k <= 1, so tau must leave at most one
    # near-miss at or above it (cosines 0.995 and 0.99 would be the top two).
    assert result.calibration_accepted <= 1
    assert result.calibration_upper <= 0.05
    previous = [c for c in result.table if c.tau < result.tau]
    assert previous and all(c.upper > 0.05 for c in previous)


def test_guards_lower_the_threshold_and_raise_recall():
    near = [
        ScoredPair("near_miss", "number", 0.5 + 0.005 * i, ("number_mismatch",)) for i in range(100)
    ]
    para = [ScoredPair("paraphrase", None, 0.4 + 0.01 * i, ()) for i in range(60)]
    result = calibrate({"calibration": near + para, "heldout": near + para}, 0.05)
    assert result.tau == min(default_grid())
    assert result.heldout.tau_unguarded > result.tau
    assert result.heldout.recall_guarded > result.heldout.recall_unguarded_calibrated


def test_unreachable_target_is_an_error():
    near = [ScoredPair("near_miss", "number", 1.0, ()) for _ in range(20)]
    with pytest.raises(ValueError):
        calibrate({"calibration": near, "heldout": near}, 0.01)


def test_calibration_on_seeded_data(calibration):
    assert calibration.calibration_upper <= 0.01
    held = calibration.heldout
    assert held.fhr_guarded <= 0.015
    assert held.recall_guarded >= 0.5
    assert held.recall_guarded > held.recall_unguarded_calibrated
    assert held.tau_unguarded > calibration.tau


def test_guard_ablation_on_seeded_data(calibration):
    by_subset = {a.subset: a for a in calibration.ablation}
    for subset in ("number", "negation"):
        a = by_subset[subset]
        assert a.accepted_all_guards == 0
        assert a.fhr_no_guards >= 5 * max(calibration.heldout.fhr_guarded, 0.01)
    assert by_subset["object"].accepted_all_guards <= by_subset["object"].accepted_no_guards
