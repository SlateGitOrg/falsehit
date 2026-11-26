"""The cache gateway: rate limit, bypass, exact and semantic lookup, guards, upstream, ledger."""

from __future__ import annotations

import hashlib
import itertools
import threading
from dataclasses import dataclass, field
from decimal import Decimal

from .clock import Clock, SystemClock, month_of
from .config import Settings
from .embed import HashingEmbedder
from .guards import Vocabulary, signature, veto
from .keys import CacheKeys, build_keys, bypass_reason
from .ledger import Ledger, LedgerRecord, Outcome, PriceTable
from .limits import RateLimiter, over_budget
from .schemas import ChatCompletionResponse, ChatRequest, Choice, ResponseMessage, Usage
from .store import Entry, SemanticStore
from .upstream import ChatModel, Completion, UpstreamError

MAX_TTL_SECONDS = 30 * 24 * 3600


class GatewayError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


@dataclass
class GatewayResult:
    response: ChatCompletionResponse
    outcome: Outcome
    request_id: str
    cosine: float | None = None
    veto: str | None = None
    bypass_reason: str | None = None
    entry: Entry | None = field(default=None, repr=False)

    def headers(self) -> dict[str, str]:
        out = {"x-cache": self.outcome, "x-request-id": self.request_id}
        if self.cosine is not None:
            out["x-cache-similarity"] = f"{self.cosine:.4f}"
        if self.veto:
            out["x-cache-veto"] = self.veto
        if self.bypass_reason:
            out["x-cache-bypass-reason"] = self.bypass_reason
        return out


class Gateway:
    def __init__(
        self,
        settings: Settings,
        upstream: ChatModel,
        clock: Clock | None = None,
        *,
        embedder: HashingEmbedder | None = None,
        store: SemanticStore | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.settings = settings
        self.upstream = upstream
        self.clock = clock or SystemClock()
        self.embedder = embedder or HashingEmbedder()
        self.store = store or SemanticStore(self.clock, settings.max_entries_per_partition)
        self.ledger = ledger or Ledger(settings.ledger_path)
        self.limiter = RateLimiter(self.clock)
        self.vocabulary = Vocabulary(settings.vocabulary)
        if settings.prices is not None:
            self.prices = PriceTable.from_mapping(settings.prices)
        else:
            self.prices = PriceTable.load(settings.prices_path)
        self._counter = itertools.count(1)
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def handle(self, tenant: str, request: ChatRequest) -> GatewayResult:
        cfg = self.settings.tenants[tenant]
        if request.stream:
            raise GatewayError(400, "unsupported_parameter", "stream=true is not supported")
        if request.model not in self.prices:
            raise GatewayError(400, "model_not_found", f"unknown model {request.model!r}")
        retry_after = self.limiter.acquire(tenant, cfg.rpm)
        if retry_after is not None:
            raise GatewayError(
                429,
                "rate_limit_exceeded",
                f"more than {cfg.rpm} requests per minute",
                {"Retry-After": str(retry_after), "x-cache": "bypass"},
            )
        now = self.clock.now()
        request_id = self._next_id("req")
        reason = bypass_reason(request)
        keys = build_keys(tenant, request)

        cosine: float | None = None
        vetoed: str | None = None
        if reason is None:
            entry, cosine, vetoed = self._lookup(tenant, keys)
            if entry is not None:
                return self._serve_hit(tenant, request_id, now, entry, cosine, request)

        outcome: Outcome = "bypass" if reason else "miss"
        self._check_budget(tenant, cfg.monthly_budget_usd, now, outcome)
        completion = self._call_upstream(request, outcome)
        cost = self._cost(request.model, completion)
        response_id = self._next_id("chatcmpl")
        stored: Entry | None = None
        if reason is None:
            ttl = min(request.cache_ttl or self.settings.ttl_seconds, MAX_TTL_SECONDS)
            stored = Entry(
                id=self._next_id("entry"),
                tenant=tenant,
                partition=keys.partition,
                exact_key=keys.exact,
                text=keys.text,
                content=completion.content,
                model=request.model,
                response_id=response_id,
                created=now,
                expires=now + ttl,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost=cost,
                tag=request.cache_tag,
                signature=signature(keys.text, self.vocabulary),
            )
            self.store.add(stored, self.embedder.embed(keys.text))
        self.ledger.append(
            LedgerRecord(
                request_id=request_id,
                ts=now,
                tenant=tenant,
                outcome=outcome,
                model=request.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_charged=cost,
                cost_avoided=Decimal(0),
                response_id=response_id,
            )
        )
        response = self._response(
            response_id,
            now,
            request.model,
            completion.content,
            completion.prompt_tokens,
            completion.completion_tokens,
            completion.finish_reason,
        )
        return GatewayResult(response, outcome, request_id, cosine, vetoed, reason, stored)

    def invalidate(self, tenant: str, tag: str | None) -> int:
        return self.store.invalidate(tenant, tag)

    def stats(self, tenant: str) -> dict:
        totals = self.ledger.totals(tenant)
        served = totals.requests
        month = month_of(self.clock.now())
        return {
            "tenant": tenant,
            "entries": self.store.live_count(tenant),
            "requests": served,
            "hits": totals.hits,
            "misses": totals.misses,
            "bypass": totals.bypass,
            "hit_rate": round(totals.hits / served, 4) if served else 0.0,
            "tau": self.settings.tau,
            "spend_usd": str(totals.charged),
            "avoided_usd": str(totals.avoided),
            "month": month,
            "month_spend_usd": str(self.ledger.month_spend(tenant, month)),
            "month_budget_usd": str(self.settings.tenants[tenant].monthly_budget_usd),
            "daily": {
                day: {
                    "requests": r.requests,
                    "hits": r.hits,
                    "spend_usd": str(r.charged),
                    "avoided_usd": str(r.avoided),
                }
                for day, r in self.ledger.daily(tenant).items()
            },
        }

    # ---------------------------------------------------------------- internals
    def _next_id(self, prefix: str) -> str:
        with self._counter_lock:
            n = next(self._counter)
        if prefix == "chatcmpl":
            return "chatcmpl-" + hashlib.sha256(f"falsehit:{n}".encode()).hexdigest()[:24]
        return f"{prefix}-{n:08d}"

    def _lookup(
        self, tenant: str, keys: CacheKeys
    ) -> tuple[Entry | None, float | None, str | None]:
        exact = self.store.get_exact(keys.partition, keys.exact)
        if exact is not None:
            return exact, 1.0, None
        query = self.embedder.embed(keys.text)
        candidates = self.store.candidates(keys.partition, query, self.settings.tau)
        if not candidates:
            return None, None, None
        sig = signature(keys.text, self.vocabulary)
        first_veto: str | None = None
        top_cosine = candidates[0][1]
        for entry, cos in candidates:
            assert entry.signature is not None
            verdict = veto(sig, entry.signature)
            if verdict is None:
                return entry, cos, None
            first_veto = first_veto or verdict
        return None, top_cosine, first_veto

    def _serve_hit(
        self,
        tenant: str,
        request_id: str,
        now: float,
        entry: Entry,
        cosine: float | None,
        request: ChatRequest,
    ) -> GatewayResult:
        entry.hits += 1
        self.ledger.append(
            LedgerRecord(
                request_id=request_id,
                ts=now,
                tenant=tenant,
                outcome="hit",
                model=entry.model,
                prompt_tokens=0,
                completion_tokens=0,
                cost_charged=Decimal(0),
                cost_avoided=entry.cost,
                response_id=entry.response_id,
            )
        )
        response = self._response(
            entry.response_id,
            entry.created,
            entry.model,
            entry.content,
            entry.prompt_tokens,
            entry.completion_tokens,
            "stop",
        )
        return GatewayResult(response, "hit", request_id, cosine, None, None, entry)

    def _check_budget(self, tenant: str, cap: Decimal, now: float, outcome: Outcome) -> None:
        spent = self.ledger.month_spend(tenant, month_of(now))
        if over_budget(spent, cap):
            raise GatewayError(
                429,
                "budget_exceeded",
                f"monthly spend cap of {cap} USD reached; cached answers are still served",
                {"x-cache": outcome},
            )

    def _call_upstream(self, request: ChatRequest, outcome: Outcome) -> Completion:
        try:
            return self.upstream.complete(request)
        except UpstreamError as exc:
            headers = {"x-cache": outcome}
            if exc.trips_breaker:
                raise GatewayError(502, "upstream_unavailable", str(exc), headers) from exc
            if exc.status in (401, 403):  # our credentials with the upstream, not the caller's
                raise GatewayError(502, "upstream_auth_failed", str(exc), headers) from exc
            raise GatewayError(exc.status, "upstream_rejected", str(exc), headers) from exc

    def _cost(self, requested_model: str, completion: Completion) -> Decimal:
        model = completion.model if completion.model in self.prices else requested_model
        return self.prices.cost(model, completion.prompt_tokens, completion.completion_tokens)

    @staticmethod
    def _response(
        response_id: str,
        created: float,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=response_id,
            created=int(created),
            model=model,
            choices=[
                Choice(message=ResponseMessage(content=content), finish_reason=finish_reason)  # type: ignore[arg-type]
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
