"""Seeded generator for the labelled pairs, the traffic and the tenants."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .intents import FAMILIES, INTENT_BY_ID, INTENTS, Intent, render
from .keys import normalize_text

SEED = 7
PAIRS_PER_CLASS = {"paraphrase": 1500, "number": 600, "negation": 450, "object": 450}
TRAFFIC_REQUESTS = 5000
ZIPF_EXPONENT = 1.1
HOT_TEMPERATURE = 0.8
HOT_SHARE = 0.10
TRAFFIC_DAYS = 3

TENANTS = {
    "brightdesk": {
        "share": 0.45,
        "system_prompt": (
            "You are the support assistant for BrightDesk, a project tracker. Answer in two "
            "sentences, name the menu path and never promise a date."
        ),
    },
    "cobaltly": {
        "share": 0.33,
        "system_prompt": (
            "You are CobaltBot, the help-desk agent for Cobaltly. Be brief and friendly, "
            "point to the exact setting and offer to hand over to a human."
        ),
    },
    "pinecrest": {
        "share": 0.22,
        "system_prompt": (
            "Pinecrest support assistant. Reply formally, give the steps as a short list and "
            "say which plan the feature needs."
        ),
    },
}


@dataclass(frozen=True)
class Pair:
    id: str
    split: str
    label: str
    subset: str | None
    a: str
    b: str
    intent_a: str
    intent_b: str


SPLITS = ("calibration", "heldout")


def family_split(family_id: str) -> str:
    """Whole families go to one split, alternating within each kind of near-miss.

    Calibration and held-out therefore share no intent, no intent pair and no template, so
    the held-out result is measured on pairs the threshold was never fitted on.
    """
    same_kind = [f.id for f in FAMILIES if f.kind == _FAMILY_KIND[family_id]]
    return SPLITS[same_kind.index(family_id) % 2]


_FAMILY_KIND = {f.id: f.kind for f in FAMILIES}


def _near_miss_intent_pairs(split: str) -> dict[str, list[tuple[Intent, Intent]]]:
    pairs: dict[str, list[tuple[Intent, Intent]]] = {"number": [], "negation": [], "object": []}
    for family in FAMILIES:
        if family_split(family.id) != split:
            continue
        members = [i for i in INTENTS if i.family == family.id]
        pairs[family.kind].extend(itertools.combinations(members, 2))
    return pairs


def _paraphrase_pairs(rng: random.Random, split: str, count: int) -> list[Pair]:
    intents = [i for i in INTENTS if family_split(i.family) == split]
    out: list[Pair] = []
    seen: set[frozenset[str]] = set()
    while len(out) < count:
        intent = intents[len(out) % len(intents)]
        template_a = rng.randrange(len(intent.templates))
        # Half the pairs reuse one template with different slot fills (a retyped canned phrase),
        # half use two different templates (a real rewording).
        template_b = template_a if rng.random() < 0.5 else rng.randrange(len(intent.templates))
        a, b = render(intent, rng, template_a), render(intent, rng, template_b)
        key = frozenset((normalize_text(a), normalize_text(b)))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(
            Pair(
                f"p{split[0]}{len(out):04d}", split, "paraphrase", None, a, b, intent.id, intent.id
            )
        )
    return out


def _near_miss_pairs(rng: random.Random, split: str, subset: str, count: int) -> list[Pair]:
    intent_pairs = _near_miss_intent_pairs(split)[subset]
    out: list[Pair] = []
    seen: set[frozenset[str]] = set()
    while len(out) < count:
        left, right = intent_pairs[len(out) % len(intent_pairs)]
        template_a = rng.randrange(len(left.templates))
        template_b = template_a if rng.random() < 0.5 else rng.randrange(len(left.templates))
        a, b = render(left, rng, template_a), render(right, rng, template_b)
        key = frozenset((normalize_text(a), normalize_text(b)))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Pair(
                f"{subset[0]}{split[0]}{len(out):04d}",
                split,
                "near_miss",
                subset,
                a,
                b,
                left.id,
                right.id,
            )
        )
    return out


def generate_pairs(seed: int = SEED) -> list[Pair]:
    """3,000 labelled pairs, 1,500 per split, with the splits disjoint by family."""
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for split in SPLITS:
        pairs += _paraphrase_pairs(rng, split, PAIRS_PER_CLASS["paraphrase"] // 2)
        for subset in ("number", "negation", "object"):
            pairs += _near_miss_pairs(rng, split, subset, PAIRS_PER_CLASS[subset] // 2)
    return pairs


def zipf_weights(n: int, exponent: float = ZIPF_EXPONENT) -> list[float]:
    raw = [1.0 / (rank**exponent) for rank in range(1, n + 1)]
    total = sum(raw)
    return [w / total for w in raw]


def popularity_order(rng: random.Random) -> list[Intent]:
    """Near-miss siblings sit next to each other in the ranking, so both are popular together."""
    families = list(FAMILIES)
    rng.shuffle(families)
    return [i for f in families for i in INTENTS if i.family == f.id]


def generate_traffic(seed: int = SEED, n: int = TRAFFIC_REQUESTS) -> list[dict]:
    rng = random.Random(seed + 1)
    ranked = popularity_order(rng)
    weights = zipf_weights(len(ranked))
    tenants = list(TENANTS)
    tenant_weights = [TENANTS[t]["share"] for t in tenants]
    span = TRAFFIC_DAYS * 24 * 3600
    times = sorted(rng.uniform(0, span) for _ in range(n))
    hot = set(rng.sample(range(n), round(n * HOT_SHARE)))
    records = []
    for i, t in enumerate(times):
        intent = rng.choices(ranked, weights)[0]
        records.append(
            {
                "t": round(t, 1),
                "tenant": rng.choices(tenants, tenant_weights)[0],
                "intent": intent.id,
                "tag": intent.category,
                "temperature": HOT_TEMPERATURE if i in hot else 0.0,
                "text": render(intent, rng),
            }
        )
    return records


def tenants_file() -> dict:
    return {
        name: {
            "api_key": f"fh-demo-{name}",
            "system_prompt": spec["system_prompt"],
            "rpm": 600,
            "monthly_budget_usd": "1000",
        }
        for name, spec in TENANTS.items()
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def seed_directory(out: Path, seed: int = SEED) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    pairs = generate_pairs(seed)
    traffic = generate_traffic(seed)
    write_jsonl(out / "pairs.jsonl", [p.__dict__ for p in pairs])
    write_jsonl(out / "traffic.jsonl", traffic)
    (out / "tenants.json").write_text(
        json.dumps(tenants_file(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"pairs": len(pairs), "traffic": len(traffic), "intents": len(INTENT_BY_ID)}
