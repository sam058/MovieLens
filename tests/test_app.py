"""Headless tests of the Streamlit UI (streamlit.testing.AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP = str(PROJECT_DIR / "src" / "app.py")
pytestmark = pytest.mark.skipif(
    not (PROJECT_DIR / "data" / "ml-25m" / "preprocessed_movies.csv").exists(),
    reason="processed MovieLens CSVs not built (see README)",
)


def submit(app, query, method=None, min_ratings=None):
    if method:
        app.sidebar.radio[0].set_value(method)
    if min_ratings is not None:
        app.sidebar.number_input[0].set_value(min_ratings)
    app.text_input[0].set_value(query)
    app.button[0].click()
    return app.run(timeout=300)


def test_first_load_prompts_for_a_query():
    app = AppTest.from_file(APP).run(timeout=60)
    assert not app.exception
    assert app.info[0].value == "Type a query and press Search."
    assert app.sidebar.radio[0].options == ["TF-IDF", "BM25", "Semantic", "RRF"]
    # Sidebar controls: the method selector and the minimum-ratings filter
    assert [n.label for n in app.sidebar.number_input] == ["Minimum number of ratings"]
    assert app.sidebar.number_input[0].value == 0
    assert not app.sidebar.slider and not app.sidebar.multiselect


def test_blank_query_shows_a_warning():
    app = submit(AppTest.from_file(APP).run(timeout=60), "    ")
    assert not app.exception
    assert app.warning[0].value == "Please enter a search query."
    assert not app.dataframe


@pytest.mark.parametrize("method", ["TF-IDF", "BM25", "Semantic", "RRF"])
def test_results_table(method):
    app = submit(AppTest.from_file(APP).run(timeout=60), "space adventure", method)
    assert not app.exception
    table = app.dataframe[0].value
    assert {"rank", "title", "year", "genres", "avg_rating", "num_ratings", "score", "plot"} <= set(table.columns)
    assert len(table) == 10
    if method == "RRF":
        assert {"TF-IDF rank", "BM25 rank", "Semantic rank"} <= set(table.columns)


def test_no_match_shows_a_warning():
    app = submit(AppTest.from_file(APP).run(timeout=60), "the of and", "BM25")
    assert not app.exception
    assert "No matching movies found" in app.warning[0].value



def test_min_ratings_filter():
    app = submit(AppTest.from_file(APP).run(timeout=60), "space adventure", "RRF", min_ratings=5000)
    assert not app.exception
    table = app.dataframe[0].value
    assert len(table) == 10 and (table["num_ratings"] >= 5000).all()
