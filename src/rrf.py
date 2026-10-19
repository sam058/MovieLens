
"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = sum over rankers r of  1 / (k + rank_r(d))

rank_r(d) is the 1-based position of movie d in ranker r's list; a movie a ranker
did not return contributes nothing for that ranker. Only rank order is used, so the
incomparable TF-IDF, BM25 and cosine scores need no normalization.
"""

DEFAULT_K = 60


def reciprocal_rank_fusion(ranked_lists, k=DEFAULT_K, top_k=None):
    """Fuse ranked result lists into one.

    ranked_lists: {ranker name: [{"movieId", "title", ...}, ...] best first}
    k: RRF smoothing constant (> 0); larger k flattens the advantage of top ranks.
    top_k: number of fused results to return (None = all).

    Returns [{"movieId", "title", "score", "ranks": {ranker: rank}}], best first.
    Ties are broken by the best single-ranker rank, then by movieId, so the output
    is deterministic.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")

    fused = {}
    for name, results in ranked_lists.items():
        for rank, result in enumerate(results or [], start=1):
            entry = fused.setdefault(
                result["movieId"],
                {"movieId": result["movieId"], "title": result.get("title"), "score": 0.0, "ranks": {}},
            )
            # A ranker that lists the same movie twice only counts its best rank
            if name in entry["ranks"]:
                continue
            entry["ranks"][name] = rank
            entry["score"] += 1.0 / (k + rank)

    ordered = sorted(
        fused.values(),
        key=lambda e: (-e["score"], min(e["ranks"].values()), e["movieId"]),
    )
    return ordered if top_k is None else ordered[:top_k]
