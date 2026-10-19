
"""One entry point for all four ranking methods (used by the UI and the evaluation).

The retrieval modules build their indexes when imported, so they are imported lazily:
asking only for BM25 does not load the sentence-transformer model.
"""

import importlib
from pathlib import Path

import pandas as pd

from rrf import DEFAULT_K, reciprocal_rank_fusion

PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_DIR / "data" / "ml-25m" / "movie_metadata.csv"

# Display name -> module providing search_movies(query, top_k)
RETRIEVERS = {
    "TF-IDF": "tfidf_search",
    "BM25": "bm25_search",
    "Semantic": "semantic_search",
}
METHODS = [*RETRIEVERS, "RRF"]

# How many results each ranker contributes to the fusion in the UI
DEFAULT_RRF_DEPTH = 100

_metadata = None


def get_retriever(method):
    """Return the search_movies function for TF-IDF, BM25 or Semantic."""
    if method not in RETRIEVERS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(RETRIEVERS)}")
    return importlib.import_module(RETRIEVERS[method]).search_movies


def movie_metadata():
    """Metadata index (year, genres, rating stats, plot) keyed by movieId, loaded once."""
    global _metadata
    if _metadata is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"{METADATA_PATH} not found. Run src/preprocessing.py first (see README).")
        metadata = pd.read_csv(METADATA_PATH, dtype={"plot": str}, keep_default_na=False, na_values={"year": [""], "avg_rating": [""]})
        # "Adventure|Comedy" -> "Adventure, Comedy" for display
        metadata["genres"] = metadata["genres"].str.replace("|", ", ", regex=False).replace({"": "(no genres listed)"})
        _metadata = metadata.set_index("movieId")
    return _metadata


def search(query, method="RRF", top_k=10, rrf_k=DEFAULT_K, rrf_depth=DEFAULT_RRF_DEPTH, min_ratings=0):
    """Rank movies for query with one method, adding metadata to each result.

    min_ratings keeps only movies with at least that many MovieLens ratings.

    Returns [{"rank", "movieId", "title", "score", "genres", "year", "avg_rating",
    "num_ratings", "plot"}] best first; RRF results also carry "ranks" ({method: rank}).
    Empty or whitespace-only queries return [].
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")

    metadata = movie_metadata()
    allowed = set(metadata.index[metadata["num_ratings"] >= min_ratings]) if min_ratings else None

    def ranked(name, count):
        if allowed is None:
            return get_retriever(name)(query, top_k=count)
        # Filter the full ranking so the result is still that method's true top-k
        results = [r for r in get_retriever(name)(query, top_k=None) if r["movieId"] in allowed]
        return results if count is None else results[:count]

    if method == "RRF":
        ranked_lists = {name: ranked(name, rrf_depth) for name in RETRIEVERS}
        results = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_k=top_k)
    else:
        results = ranked(method, top_k)

    enriched = []
    for rank, result in enumerate(results, start=1):
        info = metadata.loc[result["movieId"]] if result["movieId"] in metadata.index else None
        enriched.append({
            "rank": rank,
            **result,
            "genres": info["genres"] if info is not None else "(unknown)",
            "year": None if info is None or pd.isna(info["year"]) else int(info["year"]),
            "avg_rating": None if info is None or pd.isna(info["avg_rating"]) else float(info["avg_rating"]),
            "num_ratings": 0 if info is None else int(info["num_ratings"]),
            "plot": "" if info is None else info["plot"],
        })
    return enriched
