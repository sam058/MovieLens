import math

import pytest

import metrics

RANKED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RELEVANT = {1, 3, 6, 20}  # 20 is never retrieved


def test_precision_and_recall_at_k():
    assert metrics.precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert metrics.precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(3 / 10)
    assert metrics.recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 4)
    assert metrics.recall_at_k(RANKED, RELEVANT, 10) == pytest.approx(3 / 4)


def test_precision_divides_by_k_for_short_lists():
    assert metrics.precision_at_k([1], {1}, 5) == pytest.approx(1 / 5)
    assert metrics.precision_at_k([], {1}, 5) == 0


def test_average_precision_counts_unretrieved_relevant():
    # precisions at hits: 1/1, 2/3, 3/6; divided by |R| = 4
    assert metrics.average_precision(RANKED, RELEVANT) == pytest.approx((1 + 2 / 3 + 3 / 6) / 4)
    assert metrics.average_precision([], RELEVANT) == 0
    assert metrics.average_precision(RANKED, set()) == 0


def test_ndcg():
    gains = {1: 2, 3: 1, 6: 1}
    dcg = 3 / math.log2(2) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert metrics.ndcg_at_k(RANKED, gains, 5) == pytest.approx(dcg / ideal)
    assert metrics.ndcg_at_k([1, 3, 6], gains, 5) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(RANKED, {}, 5) == 0


def test_interpolated_precision():
    curve = metrics.interpolated_precision([1, 9, 2], {1, 2})
    # recall 0.5 at precision 1, recall 1.0 at precision 2/3
    assert curve[0] == pytest.approx(1.0)
    assert curve[5] == pytest.approx(1.0)
    assert curve[6] == pytest.approx(2 / 3)
    assert curve[10] == pytest.approx(2 / 3)
    assert metrics.interpolated_precision([9], {1}).sum() == 0
