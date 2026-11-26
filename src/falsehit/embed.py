"""An offline hashing embedding: word unigrams, bigrams and character 3-5-grams in numpy.

Features are hashed with crc32 into 2**15 signed buckets (the hashing trick), weighted by
sublinear term frequency (1 + ln tf) and l2-normalized. The result is kept sparse, because a
dense 2**15 float32 row is 128 KB and a partition can hold thousands of entries.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .keys import normalize_text

DIM = 1 << 15
CHAR_NGRAMS = (3, 4, 5)

# Function words that add shared boilerplate to every support question. Negations and
# on/off are deliberately absent: they change meaning.
STOPWORDS = frozenset(
    "a an the i me my our we us you your do does did how can could would should to of for in "
    "is are am be it its this that and or please thanks thank hi hello hey there what where "
    "when which who on at by with from as so if just also".split()
)
STOPWORDS = STOPWORDS - {"on"}


@dataclass(frozen=True)
class SparseVec:
    """A unit-length sparse vector: sorted unique bucket ids and their weights."""

    indices: np.ndarray
    values: np.ndarray

    def to_dense(self, dim: int = DIM) -> np.ndarray:
        dense = np.zeros(dim, dtype=np.float32)
        dense[self.indices] = self.values
        return dense


def content_tokens(text: str) -> list[str]:
    tokens = normalize_text(text).split()
    content = [t for t in tokens if t not in STOPWORDS]
    return content or tokens or ["<empty>"]


def _features(tokens: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in tokens:
        counts["u:" + token] += 1
    for left, right in zip(tokens, tokens[1:], strict=False):
        counts["b:" + left + " " + right] += 1
    padded = " " + " ".join(tokens) + " "
    for n in CHAR_NGRAMS:
        for start in range(len(padded) - n + 1):
            counts["c:" + padded[start : start + n]] += 1
    return counts


def _embed(text: str, dim: int) -> SparseVec:
    buckets: dict[int, float] = {}
    for feature, tf in _features(content_tokens(text)).items():
        h = zlib.crc32(feature.encode("utf-8"))
        bucket = (h & (dim - 1)) if dim & (dim - 1) == 0 else h % dim
        sign = 1.0 if (h >> 31) & 1 else -1.0
        buckets[bucket] = buckets.get(bucket, 0.0) + sign * (1.0 + math.log(tf))
    ids = np.fromiter(sorted(buckets), dtype=np.int32, count=len(buckets))
    vals = np.array([buckets[i] for i in ids.tolist()], dtype=np.float64)
    keep = vals != 0.0
    ids, vals = ids[keep], vals[keep]
    norm = float(np.linalg.norm(vals))
    if norm == 0.0:
        return SparseVec(np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))
    return SparseVec(ids, (vals / norm).astype(np.float32))


class HashingEmbedder:
    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self._cached = lru_cache(maxsize=65536)(lambda text: _embed(text, self.dim))

    def embed(self, text: str) -> SparseVec:
        return self._cached(text)


def cosine(a: SparseVec, b: SparseVec) -> float:
    """Cosine of two unit vectors, i.e. their sparse dot product."""
    _, ia, ib = np.intersect1d(a.indices, b.indices, assume_unique=True, return_indices=True)
    if ia.size == 0:
        return 0.0
    return float(np.dot(a.values[ia], b.values[ib]))
