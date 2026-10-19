
"""Ranked-retrieval metrics (TREC definitions).

ranked: list of movieIds, best first.
relevant: set of relevant movieIds (binary relevance).
gains: {movieId: graded relevance} for nDCG (missing ids have gain 0).
"""

import math

import numpy as np

RECALL_LEVELS = np.linspace(0.0, 1.0, 11)


def precision_at_k(ranked, relevant, k):
    """Relevant results in the top k divided by k (a short list is not rewarded)."""
    return sum(1 for movie in ranked[:k] if movie in relevant) / k


def recall_at_k(ranked, relevant, k):
    """Relevant results in the top k divided by all relevant movies."""
    if not relevant:
        return 0.0
    return sum(1 for movie in ranked[:k] if movie in relevant) / len(relevant)


def average_precision(ranked, relevant):
    """Mean of precision@i over the ranks i of relevant results, divided by |relevant|.

    Relevant movies that were never retrieved count as precision 0. MAP is the mean
    of this over queries.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, movie in enumerate(ranked, start=1):
        if movie in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def dcg(gain_values):
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gain_values, start=1))


def ndcg_at_k(ranked, gains, k):
    """Normalized discounted cumulative gain with exponential gain 2^g - 1."""
    ideal = dcg(sorted((g for g in gains.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([gains.get(movie, 0) for movie in ranked[:k]]) / ideal


def interpolated_precision(ranked, relevant, levels=RECALL_LEVELS):
    """11-point interpolated precision: at each recall level r, the highest precision
    reached at any recall >= r in the ranking (0 if r is never reached)."""
    precisions, recalls = [], []
    hits = 0
    for i, movie in enumerate(ranked, start=1):
        if movie in relevant:
            hits += 1
            precisions.append(hits / i)
            recalls.append(hits / len(relevant))
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    return np.array([
        precisions[recalls >= level].max() if (recalls >= level).any() else 0.0
        for level in levels
    ])
