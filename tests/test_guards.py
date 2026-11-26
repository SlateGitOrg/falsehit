import pytest

from falsehit.guards import Vocabulary, all_vetoes, signature, veto
from falsehit.intents import INTENTS, render

VOCAB = Vocabulary()


def verdict(a, b, enabled=None):
    kwargs = {} if enabled is None else {"enabled": enabled}
    return veto(signature(a, VOCAB), signature(b, VOCAB), **kwargs)


@pytest.mark.parametrize(
    "a,b",
    [
        ("export invoices for 2024", "export invoices for 2025"),
        ("refund within 14 days", "refund within 30 days"),
        ("it costs $50 a month", "it costs $500 a month"),
        ("limit of 1,000 requests", "limit of 100 requests"),
    ],
)
def test_number_mismatch(a, b):
    assert verdict(a, b) == "number_mismatch"


def test_same_numbers_in_any_format_pass():
    assert verdict("a limit of 1,000 requests", "limit 1000 requests") is None
    assert verdict("it costs $50", "50 dollars") is None
    assert signature("enable 2FA", VOCAB).numbers == frozenset()


@pytest.mark.parametrize(
    "a,b",
    [
        ("how do I enable SSO", "how do I disable SSO"),
        ("turn on auto-renewal", "turn off auto-renewal"),
        ("upgrade my plan", "downgrade my plan"),
        ("lock this account", "unlock this account"),
        ("allow guest access", "block guest access"),
        ("can I use SSO", "I can't use SSO"),
    ],
)
def test_negation_mismatch(a, b):
    assert verdict(a, b) == "negation_mismatch"


@pytest.mark.parametrize(
    "a,b",
    [
        ("Pro plan pricing", "Team plan pricing"),
        ("cancel my subscription", "cancel one seat"),
        ("reset my password", "reset the API key"),
        ("delete my account", "delete my workspace"),
    ],
)
def test_object_mismatch(a, b):
    assert verdict(a, b) == "object_mismatch"


def test_unlisted_nouns_are_not_vetoed():
    assert verdict("invite a teammate", "invite a guest") is None


def test_guards_can_be_switched_off_individually():
    a, b = "export invoices for 2024", "export invoices for 2025"
    assert verdict(a, b, enabled=("negation_mismatch", "object_mismatch")) is None
    assert all_vetoes(signature(a, VOCAB), signature(b, VOCAB)) == ("number_mismatch",)


def test_vocabulary_is_configurable():
    custom = Vocabulary({"object": {"teammate": ["teammate"], "guest": ["guest"]}})
    a, b = signature("invite a teammate", custom), signature("invite a guest", custom)
    assert veto(a, b) == "object_mismatch"


def test_every_render_of_an_intent_has_the_same_signature():
    """Paraphrases share the intent's key terms, so a guard never vetoes a true paraphrase."""
    import random

    rng = random.Random(3)
    for intent in INTENTS:
        sigs = {signature(render(intent, rng), VOCAB) for _ in range(60)}
        assert len(sigs) == 1, intent.id
