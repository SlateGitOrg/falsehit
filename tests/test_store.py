import itertools
from decimal import Decimal

from falsehit.clock import ManualClock
from falsehit.embed import HashingEmbedder
from falsehit.guards import Vocabulary, signature
from falsehit.store import Entry, SemanticStore

_COUNTER = itertools.count(1)
_COUNTER = itertools.count(1)
EMB = HashingEmbedder()
VOCAB = Vocabulary()


def entry(text, *, tenant="alpha", partition="p1", tag=None, now=0.0, ttl=100.0):
    n = next(_COUNTER)
    return Entry(
        id=f"e{n}",
        tenant=tenant,
        partition=partition,
        exact_key=f"k{n}",
        text=text,
        content="answer",
        model="m",
        response_id=f"r{n}",
        created=now,
        expires=now + ttl,
        prompt_tokens=10,
        completion_tokens=20,
        cost=Decimal("0.001"),
        tag=tag,
        signature=signature(text, VOCAB),
    )


def add(store, text, **kw):
    e = entry(text, **kw)
    store.add(e, EMB.embed(text))
    return e


def test_candidates_rank_by_cosine_within_a_partition():
    store = SemanticStore(ManualClock(0))
    near = add(store, "how do i export invoices for 2024")
    add(store, "reset my password")
    found = store.candidates("p1", EMB.embed("export invoices 2024"), 0.5)
    assert [e.id for e, _ in found] == [near.id]
    assert found[0][1] > 0.8


def test_partitions_are_isolated():
    store = SemanticStore(ManualClock(0))
    add(store, "export invoices 2024", partition="p1")
    assert store.candidates("p2", EMB.embed("export invoices 2024"), 0.1) == []


def test_ttl_expiry_under_injected_clock():
    clock = ManualClock(0)
    store = SemanticStore(clock)
    e = add(store, "export invoices 2024", ttl=100)
    query = EMB.embed("export invoices 2024")
    clock.set(99.9)
    assert store.candidates("p1", query, 0.9) and store.get_exact("p1", e.exact_key)
    clock.set(100.0)
    assert store.candidates("p1", query, 0.9) == []
    assert store.get_exact("p1", e.exact_key) is None
    assert store.live_count("alpha") == 0
    assert store.purge_expired() == 1 and len(store) == 0


def test_invalidate_by_tenant_and_tag():
    store = SemanticStore(ManualClock(0))
    add(store, "export invoices 2024", tag="billing")
    add(store, "reset my password", tag="account")
    add(store, "export invoices 2025", tag="billing", tenant="beta", partition="p2")
    assert store.invalidate("alpha", "billing") == 1
    assert store.live_count("alpha") == 1 and store.live_count("beta") == 1
    assert store.invalidate("alpha") == 1
    assert store.live_count("alpha") == 0 and store.live_count("beta") == 1


def test_snapshot_roundtrip(tmp_path):
    clock = ManualClock(0)
    store = SemanticStore(clock)
    add(store, "export invoices 2024", tag="billing", ttl=50)
    add(store, "reset my password", ttl=500)
    clock.set(60)
    path = tmp_path / "snap.jsonl"
    assert store.snapshot(path) == 1  # the expired entry is not written
    fresh = SemanticStore(clock)
    assert fresh.load(path, EMB.embed, lambda t: signature(t, VOCAB)) == 1
    found = fresh.candidates("p1", EMB.embed("reset my password"), 0.9)
    assert found and found[0][0].cost == Decimal("0.001")


def test_max_entries_evicts_oldest():
    clock = ManualClock(0)
    store = SemanticStore(clock, max_entries=3)
    for i in range(4):
        clock.set(i)
        add(store, f"question number {i} about invoices", now=float(i), ttl=1000)
    assert len(store) == 3
