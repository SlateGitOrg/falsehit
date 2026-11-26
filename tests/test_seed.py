import collections
import random

from falsehit.intents import FAMILIES, INTENT_BY_ID, INTENTS, classify, render
from falsehit.seed import generate_pairs, generate_traffic, zipf_weights


def test_catalogue_shape():
    assert len(INTENTS) == 40
    for intent in INTENTS:
        assert 8 <= len(intent.templates) <= 15, intent.id
        assert all("{k}" in t for t in intent.templates)
    assert len({i.answer for i in INTENTS}) == 40
    assert collections.Counter(f.kind for f in FAMILIES) == {
        "number": 6,
        "negation": 7,
        "object": 6,
    }


def test_keyword_table_recovers_every_intent():
    rng = random.Random(5)
    for intent in INTENTS:
        for _ in range(80):
            text = render(intent, rng)
            assert classify(text) == intent.id, text


def test_unrelated_text_is_unclassified():
    assert classify("what is the weather like") is None


def test_pair_counts_and_split():
    pairs = generate_pairs()
    assert len(pairs) == 3000
    counts = collections.Counter((p.label, p.subset) for p in pairs)
    assert counts == {
        ("paraphrase", None): 1500,
        ("near_miss", "number"): 600,
        ("near_miss", "negation"): 450,
        ("near_miss", "object"): 450,
    }
    assert collections.Counter(p.split for p in pairs) == {"calibration": 1500, "heldout": 1500}
    for split in ("calibration", "heldout"):
        assert sum(p.label == "near_miss" for p in pairs if p.split == split) == 750


def test_pair_labels_match_intents():
    for p in generate_pairs():
        if p.label == "paraphrase":
            assert p.intent_a == p.intent_b
            assert classify(p.a) == classify(p.b) == p.intent_a
        else:
            assert p.intent_a != p.intent_b
            assert INTENT_BY_ID[p.intent_a].family == INTENT_BY_ID[p.intent_b].family
            assert INTENT_BY_ID[p.intent_a].kind == p.subset


def test_generation_is_deterministic():
    assert generate_pairs(7) == generate_pairs(7)
    assert generate_traffic(7, 300) == generate_traffic(7, 300)
    assert generate_traffic(7, 300) != generate_traffic(8, 300)


def test_traffic_properties():
    traffic = generate_traffic()
    assert len(traffic) == 5000
    assert len({r["tenant"] for r in traffic}) == 3
    assert sum(r["temperature"] == 0.8 for r in traffic) == 500
    assert [r["t"] for r in traffic] == sorted(r["t"] for r in traffic)
    counts = collections.Counter(r["intent"] for r in traffic).most_common()
    assert counts[0][1] > 5 * counts[-1][1]  # a heavy head, as under Zipf


def test_zipf_weights_follow_the_exponent():
    w = zipf_weights(40, 1.1)
    assert abs(sum(w) - 1) < 1e-12
    assert abs(w[0] / w[3] - 4**1.1) < 1e-9


def test_splits_share_no_family_intent_or_template():
    pairs = generate_pairs()
    families = {
        split: {
            INTENT_BY_ID[i].family
            for p in pairs
            if p.split == split
            for i in (p.intent_a, p.intent_b)
        }
        for split in ("calibration", "heldout")
    }
    assert families["calibration"].isdisjoint(families["heldout"])
    assert len(families["calibration"]) + len(families["heldout"]) == len(FAMILIES)
    texts = {
        split: {t for p in pairs if p.split == split for t in (p.a, p.b)}
        for split in ("calibration", "heldout")
    }
    assert texts["calibration"].isdisjoint(texts["heldout"])
    for kind in ("number", "negation", "object"):
        assert {f.id for f in FAMILIES if f.kind == kind and f.id in families["calibration"]}
        assert {f.id for f in FAMILIES if f.kind == kind and f.id in families["heldout"]}
