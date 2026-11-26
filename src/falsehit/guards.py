"""Vetoes applied after a cosine match.

Cosine similarity over hashed text cannot tell "2024" from "2025", "enable" from "disable" or
"Pro" from "Team": those pairs differ in one token out of a handful. Each guard extracts a small
signature from a message and vetoes a match when the two signatures differ.

The rules are conservative on purpose. A difference in either direction vetoes, so a query that
mentions an extra number or plan simply misses the cache, which costs one upstream call, whereas
a false hit serves a wrong answer.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .keys import normalize_text

NUMBER_MISMATCH = "number_mismatch"
NEGATION_MISMATCH = "negation_mismatch"
OBJECT_MISMATCH = "object_mismatch"
ALL_GUARDS = (NUMBER_MISMATCH, NEGATION_MISMATCH, OBJECT_MISMATCH)

_NUMBER = re.compile(r"(?<!\w)\$?(\d+(?:[.,]\d+)*)%?(?!\w)")

NEGATION_WORDS = frozenset(
    "not no never without cant cannot dont doesnt didnt wont isnt arent wasnt unable none "
    "nothing neither nor".split()
)

# Antonym pairs. Each pole is a list of regular expressions over normalized text.
ANTONYM_PAIRS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "enable": (
        (
            r"enabl(?:e|es|ed|ing)",
            r"turn(?:s|ed|ing)? on",
            r"switch(?:es|ed|ing)? on",
            r"activat(?:e|es|ed|ing)",
        ),
        (
            r"disabl(?:e|es|ed|ing)",
            r"turn(?:s|ed|ing)? off",
            r"switch(?:es|ed|ing)? off",
            r"deactivat(?:e|es|ed|ing)",
        ),
    ),
    "allow": (
        (r"allow(?:s|ed|ing)?", r"permit(?:s|ted|ting)?"),
        (r"block(?:s|ed|ing)?", r"den(?:y|ies|ied|ying)", r"forbid(?:s|ding)?"),
    ),
    "upgrade": (
        (r"upgrad(?:e|es|ed|ing)",),
        (r"downgrad(?:e|es|ed|ing)",),
    ),
    "lock": (
        (r"lock(?:s|ed|ing)?", r"freez(?:e|es|ing)", r"suspend(?:s|ed|ing)?"),
        (r"unlock(?:s|ed|ing)?", r"unfreez(?:e|es|ing)", r"reinstat(?:e|es|ed|ing)"),
    ),
    "add": (
        (r"add(?:s|ed|ing)?",),
        (r"remov(?:e|es|ed|ing)", r"delet(?:e|es|ed|ing)", r"eras(?:e|es|ed|ing)"),
    ),
}

# Plans and objects whose confusion changes the answer. Keys are canonical names; values are
# aliases. Deployments extend this list; it deliberately omits some nouns (see README).
DEFAULT_VOCABULARY: dict[str, dict[str, list[str]]] = {
    "plan": {
        "starter": ["starter"],
        "pro": ["pro"],
        "team": ["team plan", "team tier"],
        "enterprise": ["enterprise"],
        "free": ["free plan", "free tier"],
    },
    "object": {
        "subscription": ["subscription"],
        "seat": ["seat", "license"],
        "workspace": ["workspace"],
        "account": ["account"],
        "password": ["password", "passcode"],
        "api_key": ["api key", "api token"],
    },
}


def _compile(patterns: Collection[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b")


class Vocabulary:
    """Configured plan and object terms, compiled for matching."""

    def __init__(self, terms: Mapping[str, Mapping[str, list[str]]] | None = None) -> None:
        self.terms = DEFAULT_VOCABULARY if terms is None else terms
        self._compiled = {
            (category, name): _compile([re.escape(a) + r"s?" for a in aliases])
            for category, group in self.terms.items()
            for name, aliases in group.items()
        }

    def find(self, normalized: str) -> frozenset[str]:
        return frozenset(
            f"{category}:{name}"
            for (category, name), pattern in self._compiled.items()
            if pattern.search(normalized)
        )


_POLES = {pair: (_compile(pos), _compile(neg)) for pair, (pos, neg) in ANTONYM_PAIRS.items()}


def _clean_number(raw: str) -> str:
    text = raw.replace(",", "")
    try:
        return format(Decimal(text).normalize(), "f")
    except InvalidOperation:
        return raw


@dataclass(frozen=True)
class Signature:
    numbers: frozenset[str]
    negation: frozenset[str]
    objects: frozenset[str]


def signature(text: str, vocabulary: Vocabulary) -> Signature:
    normalized = normalize_text(text)
    numbers = frozenset(_clean_number(m.group(1)) for m in _NUMBER.finditer(normalized))
    negation = set()
    if any(token in NEGATION_WORDS for token in normalized.split()):
        negation.add("not")
    for pair, (positive, negative) in _POLES.items():
        if positive.search(normalized):
            negation.add(f"{pair}:+")
        if negative.search(normalized):
            negation.add(f"{pair}:-")
    return Signature(numbers, frozenset(negation), vocabulary.find(normalized))


def veto(query: Signature, cached: Signature, enabled: Collection[str] = ALL_GUARDS) -> str | None:
    """The name of the first guard that rejects this pair, or None to allow the hit."""
    if NUMBER_MISMATCH in enabled and query.numbers != cached.numbers:
        return NUMBER_MISMATCH
    if NEGATION_MISMATCH in enabled and query.negation != cached.negation:
        return NEGATION_MISMATCH
    if OBJECT_MISMATCH in enabled and query.objects != cached.objects:
        return OBJECT_MISMATCH
    return None


def all_vetoes(query: Signature, cached: Signature) -> tuple[str, ...]:
    """Every guard that would reject this pair, for ablations."""
    return tuple(g for g in ALL_GUARDS if veto(query, cached, (g,)) is not None)
