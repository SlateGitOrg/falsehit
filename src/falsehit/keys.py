"""Cache keys: a tenant-and-prompt partition key, an exact key, and bypass rules."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .schemas import ChatMessage, ChatRequest

BYPASS_TEMPERATURE = 0.3
DEFAULT_TEMPERATURE = 0.0


def canonical_json(value: Any) -> str:
    """Sorted keys, no insignificant whitespace, so key order and spacing never matter."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_APOSTROPHES = re.compile(r"[’'`]")
_SYMBOLS = re.compile(r"[^\w\s$%.,]")
_LOOSE_PUNCT = re.compile(r"(?<!\d)[.,]|[.,](?!\d)")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Casefold, drop apostrophes, turn punctuation into spaces, collapse whitespace.

    Dots, commas, `$` and `%` survive only when they are part of a number, so "$1,200.50"
    and "10%" keep their meaning while "invoices, 2024." becomes "invoices 2024".
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _APOSTROPHES.sub("", text)
    text = _SYMBOLS.sub(" ", text)
    text = _LOOSE_PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def system_prompt_of(messages: list[ChatMessage]) -> str:
    return "\n".join(m.content for m in messages if m.role in ("system", "developer"))


def last_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def effective_temperature(request: ChatRequest) -> float:
    """A request that omits temperature is pinned to 0 so it is cacheable and deterministic."""
    return float(DEFAULT_TEMPERATURE if request.temperature is None else request.temperature)


def partition_key(
    *,
    tenant: str,
    model: str,
    system_prompt: str,
    temperature: float,
    top_p: float | None,
    tools: list[dict[str, Any]] | None,
    response_format: dict[str, Any] | None,
    max_tokens: int | None = None,
) -> str:
    """sha256 over canonical JSON of everything two requests must share to share answers."""
    payload = {
        "tenant": tenant,
        "model": model,
        "system": sha256_hex(system_prompt),
        "temperature": float(temperature),
        "top_p": None if top_p is None else float(top_p),
        "tools": tools,
        "response_format": response_format,
        "max_tokens": max_tokens,
    }
    return sha256_hex(canonical_json(payload))


def exact_key(partition: str, last_user: str) -> str:
    return sha256_hex(partition + "\n" + normalize_text(last_user))


@dataclass(frozen=True)
class CacheKeys:
    partition: str
    exact: str
    text: str
    """The normalized last user message."""


def build_keys(tenant: str, request: ChatRequest) -> CacheKeys:
    partition = partition_key(
        tenant=tenant,
        model=request.model,
        system_prompt=system_prompt_of(request.messages),
        temperature=effective_temperature(request),
        top_p=request.top_p,
        tools=request.tools,
        response_format=request.response_format,
        max_tokens=request.max_tokens,
    )
    last_user = last_user_message(request.messages)
    return CacheKeys(partition, exact_key(partition, last_user), normalize_text(last_user))


def bypass_reason(request: ChatRequest) -> str | None:
    """Why this request must skip the cache, or None when it may use it."""
    if not request.cache:
        return "cache_disabled"
    if request.tools:
        return "tools"
    if effective_temperature(request) > BYPASS_TEMPERATURE:
        return "temperature"
    # The exact key covers only the last user message, so any earlier turn would be ignored.
    if any(m.role in ("assistant", "tool") for m in request.messages):
        return "multi_turn"
    if sum(m.role == "user" for m in request.messages) > 1:
        return "multi_turn"
    return None
