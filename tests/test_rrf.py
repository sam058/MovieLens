import pytest

from rrf import DEFAULT_K, reciprocal_rank_fusion


def results(*ids):
    return [{"movieId": i, "title": f"Movie {i}", "score": 1.0} for i in ids]


def test_standard_formula_with_default_k():
    fused = reciprocal_rank_fusion({"a": results(1, 2, 3), "b": results(3, 1)})
    scores = {r["movieId"]: r["score"] for r in fused}
    assert DEFAULT_K == 60
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert [r["movieId"] for r in fused] == [1, 3, 2]
    assert fused[0]["ranks"] == {"a": 1, "b": 2}


def test_configurable_k_changes_order():
    # Movie 1 is first in one list; movie 2 is fourth in both lists
    lists = {"a": results(1, 7, 8, 2), "b": results(5, 6, 9, 2)}
    # k=1: 1/2 = 0.5 beats 2/5 = 0.4 -> a single top rank wins
    assert reciprocal_rank_fusion(lists, k=1)[0]["movieId"] in (1, 5)
    # k=60: 2/64 beats 1/61 -> agreement between lists wins
    assert reciprocal_rank_fusion(lists, k=60)[0]["movieId"] == 2


def test_top_k_empty_lists_and_duplicates():
    assert reciprocal_rank_fusion({"a": [], "b": None}) == []
    fused = reciprocal_rank_fusion({"a": results(1, 1, 2)}, top_k=1)
    assert len(fused) == 1 and fused[0]["score"] == pytest.approx(1 / 61)


def test_ties_are_deterministic():
    fused = reciprocal_rank_fusion({"a": results(5, 7), "b": results(7, 5)})
    assert fused[0]["score"] == pytest.approx(fused[1]["score"])
    assert [r["movieId"] for r in fused] == [5, 7]


def test_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": results(1)}, k=0)
