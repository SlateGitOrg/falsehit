import json

from falsehit.keys import (
    bypass_reason,
    canonical_json,
    exact_key,
    normalize_text,
    partition_key,
)
from falsehit.schemas import ChatRequest

BASE = dict(
    tenant="alpha",
    model="m",
    system_prompt="be brief",
    temperature=0.0,
    top_p=None,
    tools=None,
    response_format={"type": "json_object", "schema": {"b": 1, "a": 2}},
)


def test_canonical_json_ignores_key_order_and_whitespace():
    one = canonical_json(json.loads('{"b": 1,  "a": {"y": 2, "x": 3}}'))
    two = canonical_json(json.loads('{"a":{"x":3,"y":2},"b":1}'))
    assert one == two == '{"a":{"x":3,"y":2},"b":1}'


def test_partition_key_ignores_response_format_key_order():
    reordered = {**BASE, "response_format": {"schema": {"a": 2, "b": 1}, "type": "json_object"}}
    assert partition_key(**BASE) == partition_key(**reordered)


def test_partition_key_changes_with_each_ingredient():
    base = partition_key(**BASE)
    for change in (
        {"system_prompt": "be verbose"},
        {"temperature": 0.2},
        {"model": "other"},
        {"tenant": "beta"},
        {"top_p": 0.9},
        {"max_tokens": 20},
        {"tools": [{"type": "function"}]},
    ):
        assert partition_key(**{**BASE, **change}) != base, change


def test_exact_key_normalizes_case_space_and_punctuation():
    p = partition_key(**BASE)
    assert exact_key(p, "How do I   cancel my subscription?") == exact_key(
        p, "how do i cancel my subscription"
    )
    assert exact_key(p, "cancel one seat") != exact_key(p, "cancel my subscription")


def test_normalize_text_keeps_numbers_intact():
    assert normalize_text("Export invoices, 2024.") == "export invoices 2024"
    assert normalize_text("It costs $1,200.50 (10%)!") == "it costs $1,200.50 10%"
    assert normalize_text("Can’t log in") == "cant log in"


def _req(**kw):
    return ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], **kw)


def test_bypass_rules():
    assert bypass_reason(_req()) is None
    assert bypass_reason(_req(temperature=0.3)) is None
    assert bypass_reason(_req(temperature=0.31)) == "temperature"
    assert bypass_reason(_req(tools=[{"type": "function"}])) == "tools"
    assert bypass_reason(_req(cache=False)) == "cache_disabled"
    multi = ChatRequest(
        model="m",
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    )
    assert bypass_reason(multi) == "multi_turn"
    two_users = ChatRequest(
        model="m",
        messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    )
    assert bypass_reason(two_users) == "multi_turn"
