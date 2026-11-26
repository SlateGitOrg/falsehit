import numpy as np

from falsehit.embed import DIM, HashingEmbedder, cosine


def test_embedding_is_deterministic_and_unit_length():
    e = HashingEmbedder()
    a, b = (
        e.embed("How do I export invoices?"),
        HashingEmbedder().embed("How do I export invoices?"),
    )
    assert np.array_equal(a.indices, b.indices) and np.array_equal(a.values, b.values)
    assert abs(float(np.linalg.norm(a.values)) - 1.0) < 1e-5
    assert a.indices.max() < DIM


def test_identical_content_scores_one_and_unrelated_scores_low():
    e = HashingEmbedder()
    assert (
        abs(cosine(e.embed("export invoices 2024"), e.embed("Export invoices, 2024!")) - 1) < 1e-5
    )
    assert cosine(e.embed("export invoices 2024"), e.embed("reset my password")) < 0.15


def test_one_token_difference_is_close_but_not_identical():
    e = HashingEmbedder()
    sim = cosine(e.embed("export invoices for 2024"), e.embed("export invoices for 2025"))
    assert 0.6 < sim < 0.99


def test_empty_text_still_embeds():
    v = HashingEmbedder().embed("")
    assert v.indices.size >= 1
