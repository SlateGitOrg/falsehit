"""An in-memory semantic index per partition.

Each partition holds its entries as a CSR-style set of numpy arrays. A lookup scatters the query
into a dense vector once and scores every live entry with one gather and one segmented sum, so
it is an exact brute-force cosine over the whole partition.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np

from .clock import Clock
from .embed import DIM, SparseVec
from .guards import Signature


@dataclass
class Entry:
    id: str
    tenant: str
    partition: str
    exact_key: str
    text: str
    """Normalized last user message."""
    content: str
    model: str
    response_id: str
    created: float
    expires: float
    prompt_tokens: int
    completion_tokens: int
    cost: Decimal
    tag: str | None = None
    hits: int = 0
    signature: Signature | None = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("signature")
        data["cost"] = str(self.cost)
        return data

    @classmethod
    def from_json(cls, data: dict) -> Entry:
        return cls(**{**data, "cost": Decimal(data["cost"])})


class _Partition:
    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self.vectors: list[SparseVec] = []
        self.by_exact: dict[str, int] = {}
        self._matrix: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def add(self, entry: Entry, vector: SparseVec) -> None:
        self.by_exact[entry.exact_key] = len(self.entries)
        self.entries.append(entry)
        self.vectors.append(vector)
        self._matrix = None

    def rebuild(self, keep: Callable[[Entry], bool]) -> int:
        kept = [(e, v) for e, v in zip(self.entries, self.vectors, strict=True) if keep(e)]
        removed = len(self.entries) - len(kept)
        self.entries = [e for e, _ in kept]
        self.vectors = [v for _, v in kept]
        self.by_exact = {e.exact_key: i for i, e in enumerate(self.entries)}
        self._matrix = None
        return removed

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._matrix is None:
            sizes = np.array([v.indices.size for v in self.vectors], dtype=np.int64)
            indptr = np.concatenate(([0], np.cumsum(sizes)))
            indices = np.concatenate([v.indices for v in self.vectors])
            data = np.concatenate([v.values for v in self.vectors])
            self._matrix = (indptr, indices, data)
        return self._matrix

    def scores(self, query: SparseVec) -> np.ndarray:
        indptr, indices, data = self.matrix()
        dense = np.zeros(DIM, dtype=np.float32)
        dense[query.indices] = query.values
        return np.add.reduceat(data * dense[indices], indptr[:-1])


class SemanticStore:
    def __init__(self, clock: Clock, max_entries: int = 4096) -> None:
        self._clock = clock
        self._max = max_entries
        self._partitions: dict[str, _Partition] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ reads
    def get_exact(self, partition: str, exact_key: str) -> Entry | None:
        with self._lock:
            part = self._partitions.get(partition)
            if part is None:
                return None
            idx = part.by_exact.get(exact_key)
            if idx is None:
                return None
            entry = part.entries[idx]
            return entry if entry.expires > self._clock.now() else None

    def candidates(
        self, partition: str, query: SparseVec, tau: float, limit: int = 8
    ) -> list[tuple[Entry, float]]:
        """Live entries with cosine >= tau, best first."""
        with self._lock:
            part = self._partitions.get(partition)
            if part is None or not part.entries:
                return []
            now = self._clock.now()
            scores = part.scores(query)
            live = np.fromiter((e.expires > now for e in part.entries), dtype=bool)
            ok = np.flatnonzero(live & (scores >= tau - 1e-6))
            ordered = ok[np.argsort(-scores[ok], kind="stable")][:limit]
            return [(part.entries[i], float(scores[i])) for i in ordered]

    # ----------------------------------------------------------------- writes
    def add(self, entry: Entry, vector: SparseVec) -> None:
        with self._lock:
            part = self._partitions.setdefault(entry.partition, _Partition())
            now = self._clock.now()
            if len(part.entries) >= self._max:
                part.rebuild(lambda e: e.expires > now)
            if len(part.entries) >= self._max:
                oldest = min(e.created for e in part.entries)
                part.rebuild(lambda e: e.created > oldest)
            part.add(entry, vector)

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock.now()
            return sum(p.rebuild(lambda e: e.expires > now) for p in self._partitions.values())

    def invalidate(self, tenant: str, tag: str | None = None) -> int:
        """Drop a tenant's entries, or only those carrying `tag`."""
        with self._lock:
            return sum(
                part.rebuild(lambda e: not (e.tenant == tenant and (tag is None or e.tag == tag)))
                for part in self._partitions.values()
            )

    # ------------------------------------------------------------------ stats
    def live_count(self, tenant: str) -> int:
        with self._lock:
            now = self._clock.now()
            return sum(
                1
                for p in self._partitions.values()
                for e in p.entries
                if e.tenant == tenant and e.expires > now
            )

    def __len__(self) -> int:
        with self._lock:
            return sum(len(p.entries) for p in self._partitions.values())

    # --------------------------------------------------------------- snapshot
    def snapshot(self, path: Path) -> int:
        """Write live entries as JSONL. Vectors are not stored; they are recomputed on load."""
        with self._lock:
            now = self._clock.now()
            lines = [
                json.dumps(e.to_json(), sort_keys=True)
                for p in self._partitions.values()
                for e in p.entries
                if e.expires > now
            ]
        Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return len(lines)

    def load(
        self,
        path: Path,
        embed: Callable[[str], SparseVec],
        sign: Callable[[str], Signature],
    ) -> int:
        count = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = Entry.from_json(json.loads(line))
            entry.signature = sign(entry.text)
            self.add(entry, embed(entry.text))
            count += 1
        return count
