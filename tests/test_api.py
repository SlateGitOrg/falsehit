from conftest import Harness, make_settings

from falsehit.embed import HashingEmbedder, cosine
from falsehit.schemas import ChatCompletionResponse

Q2025 = "How do I export invoices for 2025"
Q2024 = "How do I export invoices for 2024"


def test_response_matches_openai_schema_and_carries_x_cache(harness):
    r = harness.ask(Q2025)
    assert r.status_code == 200 and r.headers["x-cache"] == "miss"
    body = ChatCompletionResponse.model_validate(r.json())
    assert body.object == "chat.completion" and body.choices[0].message.role == "assistant"
    assert body.usage.total_tokens == body.usage.prompt_tokens + body.usage.completion_tokens
    assert "2025" in body.choices[0].message.content


def test_exact_repeat_is_a_hit_with_the_original_response_id(harness):
    first = harness.ask(Q2025)
    second = harness.ask("how do i export invoices for 2025?")  # same after normalization
    assert second.headers["x-cache"] == "hit"
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["choices"] == first.json()["choices"]
    assert harness.upstream.calls == 1


def test_2024_after_cached_2025_is_a_miss_with_number_veto_despite_high_cosine(harness):
    tau = harness.gateway.settings.tau
    emb = HashingEmbedder()
    assert cosine(emb.embed(Q2024), emb.embed(Q2025)) >= tau
    harness.ask(Q2025)
    r = harness.ask(Q2024)
    assert r.headers["x-cache"] == "miss"
    assert r.headers["x-cache-veto"] == "number_mismatch"
    assert "2024" in r.json()["choices"][0]["message"]["content"]
    assert harness.upstream.calls == 2


def test_semantic_hit_for_a_paraphrase(harness):
    harness.ask("How do I export invoices for 2025")
    r = harness.ask("Where can I download all invoices from 2025?")
    assert r.headers["x-cache"] == "hit"
    assert float(r.headers["x-cache-similarity"]) >= harness.gateway.settings.tau
    assert harness.upstream.calls == 1


def test_bypass_for_temperature_tools_and_cache_false(harness):
    harness.ask(Q2025)
    for extra in (
        {"temperature": 0.8},
        {"tools": [{"type": "function", "function": {"name": "f"}}]},
        {"cache": False},
    ):
        r = harness.ask(Q2025, **extra)
        assert r.headers["x-cache"] == "bypass", extra
    assert harness.upstream.calls == 4
    # 0.3 is cacheable but is its own partition, so it misses once and then hits.
    assert harness.ask(Q2025, temperature=0.3).headers["x-cache"] == "miss"
    assert harness.ask(Q2025, temperature=0.3).headers["x-cache"] == "hit"


def test_system_prompt_and_temperature_partition_the_cache(harness):
    harness.ask(Q2025)
    assert harness.ask(Q2025, system="You are a pirate.").headers["x-cache"] == "miss"
    assert harness.ask(Q2025, temperature=0.2).headers["x-cache"] == "miss"
    assert harness.ask(Q2025, temperature=0.2).headers["x-cache"] == "hit"


def test_tenants_never_share_answers_even_with_identical_prompts(harness):
    harness.ask(Q2025, key="key-alpha")
    assert harness.ask(Q2025, key="key-beta").headers["x-cache"] == "miss"
    assert harness.ask(Q2025, key="key-beta").headers["x-cache"] == "hit"
    assert harness.upstream.calls == 2
    entries = {e.tenant for p in harness.gateway.store._partitions.values() for e in p.entries}
    assert entries == {"alpha", "beta"}


def test_ttl_expiry_through_the_api(harness):
    harness.ask(Q2025)
    harness.clock.advance(599)
    assert harness.ask(Q2025).headers["x-cache"] == "hit"
    harness.clock.advance(2)
    assert harness.ask(Q2025).headers["x-cache"] == "miss"


def test_per_request_ttl_and_invalidate_by_tag(harness):
    harness.ask(Q2025, cache_tag="billing", cache_ttl=50)
    harness.ask("how do I reset my password", cache_tag="account")
    headers = {"Authorization": "Bearer key-alpha"}
    removed = harness.client.post("/v1/cache/invalidate", json={"tag": "billing"}, headers=headers)
    assert removed.json() == {"tenant": "alpha", "tag": "billing", "removed": 1}
    assert harness.ask(Q2025).headers["x-cache"] == "miss"
    assert harness.ask("how do I reset my password").headers["x-cache"] == "hit"
    other = harness.client.post(
        "/v1/cache/invalidate", json={}, headers={"Authorization": "Bearer key-beta"}
    )
    assert other.json()["removed"] == 0
    assert (
        harness.client.post("/v1/cache/invalidate", json={}, headers=headers).json()["removed"] == 2
    )


def test_stats_endpoint_reports_the_ledger(harness):
    harness.ask(Q2025)
    harness.ask(Q2025)
    harness.ask(Q2025, temperature=0.9)
    r = harness.client.get("/v1/cache/stats", headers={"Authorization": "Bearer key-alpha"})
    s = r.json()
    assert (s["requests"], s["hits"], s["misses"], s["bypass"]) == (3, 1, 1, 1)
    assert s["hit_rate"] == 0.3333 and s["entries"] == 1
    assert float(s["avoided_usd"]) > 0 and float(s["spend_usd"]) > float(s["avoided_usd"])
    assert r.headers["x-cache"] == "bypass"


def test_auth_and_validation_errors_use_the_openai_error_shape(harness):
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic key-alpha"}):
        r = harness.client.post("/v1/chat/completions", json={}, headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
        assert "x-cache" in r.headers
    bad = harness.client.post(
        "/v1/chat/completions",
        json={"model": "support-large"},
        headers={"Authorization": "Bearer key-alpha"},
    )
    assert bad.status_code == 400 and bad.json()["error"]["type"] == "invalid_request_error"


def test_unknown_model_and_streaming_are_rejected(harness):
    body = {"model": "gpt-nope", "messages": [{"role": "user", "content": "hi"}]}
    r = harness.client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-alpha"}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_found"
    r = harness.ask(Q2025, stream=True)
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"


def test_all_upstreams_down_returns_502_and_charges_nothing():
    from falsehit.upstream import FakeUpstream

    h = Harness(upstream=FakeUpstream(fail_on={1}))
    r = h.ask(Q2025)
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_unavailable"
    assert h.gateway.ledger.totals("alpha").requests == 0


def test_settings_tenant_lookup():
    s = make_settings()
    assert s.tenant_for_key("key-beta") == "beta" and s.tenant_for_key("zzz") is None


def test_max_tokens_partitions_the_cache(harness):
    """A reply cut short by max_tokens must not be served to a request with no limit."""
    assert harness.ask(Q2025, max_tokens=20).headers["x-cache"] == "miss"
    assert harness.ask(Q2025).headers["x-cache"] == "miss"
    assert harness.ask(Q2025, max_tokens=20).headers["x-cache"] == "hit"
    assert harness.ask(Q2025, max_tokens=21).headers["x-cache"] == "miss"
    assert harness.ask(Q2025).headers["x-cache"] == "hit"


def test_consecutive_user_turns_bypass_the_cache(harness):
    def ask_two(first):
        body = {
            "model": "support-large",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": first},
                {"role": "user", "content": Q2025},
            ],
        }
        return harness.client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-alpha"}
        )

    harness.ask(Q2025)
    for first in ("about the 2024 file", "about the 2023 file"):
        r = ask_two(first)
        assert r.headers["x-cache"] == "bypass"
        assert r.headers["x-cache-bypass-reason"] == "multi_turn"
    assert harness.upstream.calls == 3


def test_upstream_client_errors_are_surfaced_not_masked():
    from falsehit.upstream import UpstreamError

    class Rejecting:
        name = "rejecting"

        def __init__(self, status):
            self.status = status

        def complete(self, request):
            raise UpstreamError(self.status, "nope")

    h = Harness(upstream=Rejecting(400))
    r = h.ask(Q2025)
    assert r.status_code == 400 and r.json()["error"]["code"] == "upstream_rejected"
    h = Harness(upstream=Rejecting(401))
    r = h.ask(Q2025)
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_auth_failed"
