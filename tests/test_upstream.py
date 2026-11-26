import json
import math
import os

import httpx
import pytest

from falsehit.intents import INTENT_BY_ID
from falsehit.schemas import ChatRequest
from falsehit.upstream import FakeUpstream, OpenAICompatUpstream, UpstreamError


def req(text, system="be brief", **kw):
    return ChatRequest(
        model="support-large",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        **kw,
    )


def test_fake_returns_the_intent_answer_with_ceil_usage():
    up = FakeUpstream()
    text = "How do I disable SSO for our company?"
    out = up.complete(req(text))
    intent = INTENT_BY_ID["sso_toggle/disable"]
    assert out.content == intent.answer
    assert out.prompt_tokens == math.ceil((len("be brief") + len(text)) / 4)
    assert out.completion_tokens == intent.completion_tokens


def test_fake_is_deterministic_and_has_a_fallback():
    up = FakeUpstream()
    assert up.complete(req("export invoices 2024")) == up.complete(req("export invoices 2024"))
    assert "not sure" in up.complete(req("tell me a joke")).content


def test_scripted_failures_raise_503_on_those_calls_only():
    up = FakeUpstream(fail_on={2})
    up.complete(req("export invoices 2024"))
    with pytest.raises(UpstreamError) as exc:
        up.complete(req("export invoices 2024"))
    assert exc.value.status == 503
    up.complete(req("export invoices 2024"))
    assert up.calls == 3


def test_openai_compat_posts_and_parses_with_a_mock_transport():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "remote-model",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

    up = OpenAICompatUpstream(
        "https://llm.example/v1/", "secret", "remote-model", transport=httpx.MockTransport(handler)
    )
    out = up.complete(req("hi", temperature=0.1))
    assert (out.content, out.prompt_tokens, out.completion_tokens) == ("hello", 7, 3)
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "remote-model" and seen["body"]["temperature"] == 0.1
    assert "cache" not in seen["body"]


def test_openai_compat_maps_errors_to_upstream_error():
    up = OpenAICompatUpstream(
        "https://llm.example", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    with pytest.raises(UpstreamError) as exc:
        up.complete(req("hi"))
    assert exc.value.status == 500
    bad = OpenAICompatUpstream(
        "https://llm.example",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": 1})),
    )
    with pytest.raises(UpstreamError) as exc:
        bad.complete(req("hi"))
    assert exc.value.status == 502


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("FALSEHIT_LIVE") != "1", reason="set FALSEHIT_LIVE=1")
def test_live_upstream_round_trip():
    out = OpenAICompatUpstream.from_env().complete(req("Reply with the single word: ok"))
    assert out.content and out.completion_tokens > 0
