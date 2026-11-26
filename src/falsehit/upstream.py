"""Upstream chat models behind one protocol: a deterministic fake and an HTTP client."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import httpx

from .intents import INTENT_BY_ID, UNKNOWN_ANSWER, UNKNOWN_COMPLETION_TOKENS, classify
from .keys import last_user_message
from .schemas import ChatRequest


class UpstreamError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def trips_breaker(self) -> bool:
        """Server faults, rate limiting and transport errors (reported as 503) count against
        the upstream. Any other 4xx means the request or the credentials were wrong, not that
        the upstream is unhealthy."""
        return self.status >= 500 or self.status == 429


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class ChatModel(Protocol):
    name: str

    def complete(self, request: ChatRequest) -> Completion: ...


class FakeUpstream:
    """Deterministic offline model.

    It maps the last user message to an intent with the generator's keyword table and returns
    that intent's canned answer. `fail_on` lists 1-based call indexes that raise a 503, which
    drives the circuit-breaker tests.
    """

    def __init__(self, name: str = "fake", fail_on: Iterable[int] = ()) -> None:
        self.name = name
        self.fail_on = frozenset(fail_on)
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, request: ChatRequest) -> Completion:
        with self._lock:
            self.calls += 1
            call = self.calls
        if call in self.fail_on:
            raise UpstreamError(503, f"{self.name} scripted failure on call {call}")
        chars = sum(len(m.content) for m in request.messages)
        intent_id = classify(last_user_message(request.messages))
        if intent_id is None:
            answer, completion = UNKNOWN_ANSWER, UNKNOWN_COMPLETION_TOKENS
        else:
            intent = INTENT_BY_ID[intent_id]
            answer, completion = intent.answer, intent.completion_tokens
        return Completion(answer, request.model, math.ceil(chars / 4), completion)


class OpenAICompatUpstream:
    """An OpenAI-compatible `/chat/completions` endpoint, configured from the environment."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str | None = None,
        name: str = "openai-compat",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = name
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def from_env(
        cls, prefix: str = "FALSEHIT_UPSTREAM", name: str = "primary"
    ) -> OpenAICompatUpstream:
        return cls(
            base_url=os.environ[f"{prefix}_BASE_URL"],
            api_key=os.environ.get(f"{prefix}_API_KEY", ""),
            model=os.environ.get(f"{prefix}_MODEL") or None,
            name=name,
        )

    def complete(self, request: ChatRequest) -> Completion:
        body = request.model_dump(
            include={"messages", "temperature", "top_p", "max_tokens", "tools", "response_format"},
            exclude_none=True,
        )
        body["model"] = self._model or request.model
        try:
            response = self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise UpstreamError(503, f"{self.name} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise UpstreamError(
                response.status_code, f"{self.name} returned {response.status_code}"
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data["usage"]
            return Completion(
                content=choice["message"]["content"] or "",
                model=data.get("model", body["model"]),
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                finish_reason=choice.get("finish_reason") or "stop",
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamError(502, f"{self.name} returned a malformed body") from exc
