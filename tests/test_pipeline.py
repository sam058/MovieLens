"""End-to-end tests over the real data. Skipped when the processed CSVs are missing.

The first run loads the sentence-transformer model and embeddings (slow on CPU).
"""

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-25m"
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "preprocessed_movies.csv").exists() or not (DATA_DIR / "processed_movies.csv").exists(),
    reason="processed MovieLens CSVs not built (see README)",
)

ALL_METHODS = ["TF-IDF", "BM25", "Semantic", "RRF"]


@pytest.fixture(scope="module")
def engine():
    import search_engine

    for name in search_engine.RETRIEVERS:
        search_engine.get_retriever(name)
    return search_engine


@pytest.fixture(scope="module")
def movie_ids():
    return set(pd.read_csv(DATA_DIR / "processed_movies.csv", usecols=["movieId"])["movieId"])


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("query", ["", "   ", "\n\t", None])
def test_empty_queries_return_nothing(engine, method, query):
    assert engine.search(query, method=method) == []


@pytest.mark.parametrize("method", ["TF-IDF", "BM25"])
@pytest.mark.parametrize("query", ["the and of", "!!! ???", "zzzqqqxxv"])
def test_keyword_methods_return_empty_for_unmatchable_queries(engine, method, query):
    assert engine.search(query, method=method) == []


@pytest.mark.parametrize("name", ["TF-IDF", "BM25", "Semantic"])
def test_base_retrievers_share_the_rrf_format(engine, name):
    results = engine.get_retriever(name)("space adventure", top_k=10)
    assert 0 < len(results) <= 10
    assert all(list(r) == ["movieId", "title", "score"] for r in results)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_search_results_are_complete_and_ranked(engine, movie_ids, method):
    results = engine.search("time travel", method=method, top_k=7)
    assert len(results) == 7
    assert [r["rank"] for r in results] == list(range(1, 8))
    assert len({r["movieId"] for r in results}) == 7
    for r in results:
        assert r["movieId"] in movie_ids
        assert isinstance(r["title"], str) and r["title"]
        assert isinstance(r["genres"], str) and r["genres"]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_obvious_title_queries_find_the_movie(engine):
    for method in ALL_METHODS:
        titles = [r["title"] for r in engine.search("toy story", method=method, top_k=10)]
        assert any(t.startswith("Toy Story") for t in titles), (method, titles)


def test_rrf_scores_match_component_ranks(engine):
    k = 60
    results = engine.search("funny space adventure", method="RRF", top_k=10, rrf_k=k, rrf_depth=50)
    for r in results:
        assert r["ranks"] and all(1 <= rank <= 50 for rank in r["ranks"].values())
        assert r["score"] == pytest.approx(sum(1 / (k + rank) for rank in r["ranks"].values()))


def test_rrf_k_is_configurable(engine):
    a = engine.search("heist", method="RRF", top_k=20, rrf_k=1)
    b = engine.search("heist", method="RRF", top_k=20, rrf_k=500)
    assert a[0]["score"] > b[0]["score"]


def test_accented_query_matches_accented_title(engine):
    titles = [r["title"] for r in engine.search("Amélie", method="BM25", top_k=5)]
    assert any("Amélie" in t for t in titles), titles


def test_invalid_arguments(engine):
    with pytest.raises(ValueError):
        engine.search("space", method="LSI")
    with pytest.raises(ValueError):
        engine.search("space", method="BM25", top_k=0)


def test_missing_genres_are_labelled(engine):
    genres = set(engine.movie_metadata()["genres"])
    assert "(no genres listed)" in genres
    assert "" not in genres


def test_results_carry_metadata(engine):
    metadata = engine.movie_metadata()
    toy_story = metadata.loc[1]
    assert toy_story["year"] == 1995
    assert toy_story["genres"] == "Adventure, Animation, Children, Comedy, Fantasy"
    assert 3.5 < toy_story["avg_rating"] < 4.5 and toy_story["num_ratings"] > 10000
    assert toy_story["plot"]

    for r in engine.search("toy story", method="RRF", top_k=10):
        row = metadata.loc[r["movieId"]]
        assert r["genres"] == row["genres"] and r["num_ratings"] == row["num_ratings"]
        assert r["plot"] == row["plot"]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_min_ratings_filter(engine, method):
    results = engine.search("heist", method=method, top_k=10, min_ratings=1000)
    assert len(results) == 10
    assert all(r["num_ratings"] >= 1000 for r in results)


def test_min_ratings_keeps_the_unfiltered_order(engine):
    unfiltered = engine.search("heist", method="BM25", top_k=None)
    filtered = engine.search("heist", method="BM25", top_k=5, min_ratings=1000)
    expected = [r["movieId"] for r in unfiltered if r["num_ratings"] >= 1000][:5]
    assert [r["movieId"] for r in filtered] == expected


def test_min_ratings_surfaces_the_well_known_movie(engine):
    # Long tag lists push Toy Story below its short spin-offs; the filter brings it up
    assert 1 not in [r["movieId"] for r in engine.search("toy story", method="BM25", top_k=5)]
    assert engine.search("toy story", method="BM25", top_k=1, min_ratings=50000)[0]["movieId"] == 1


def test_impossible_min_ratings_returns_nothing(engine):
    assert engine.search("heist", method="RRF", min_ratings=10**7) == []
