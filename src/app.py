
"""Streamlit movie search UI.  Run from the project root:  streamlit run src/app.py"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file as a script; make the sibling modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import search_engine  # noqa: E402
from rrf import DEFAULT_K  # noqa: E402

st.set_page_config(page_title="MovieLens Search", page_icon="🎬", layout="wide")
st.title("🎬 MovieLens Movie Search")
st.caption(
    "Search 62k MovieLens 25M movies by title, genre, plot and user tags with TF-IDF, BM25, "
    "sentence embeddings, or their Reciprocal Rank Fusion."
)

SCORE_HELP = {
    "TF-IDF": "cosine similarity of TF-IDF vectors (0 to 1)",
    "BM25": "BM25 relevance (0 to ∞)",
    "Semantic": "cosine similarity of all-MiniLM-L6-v2 embeddings (-1 to 1)",
    "RRF": f"sum of 1 / ({DEFAULT_K} + rank) over the three methods",
}


@st.cache_resource(show_spinner=False)
def load_method(method):
    """Import the retrieval modules once per server process (builds the indexes)."""
    names = search_engine.RETRIEVERS if method == "RRF" else [method]
    for name in names:
        search_engine.get_retriever(name)
    return True


TOP_K = 10  # matches the P@10 / nDCG@10 evaluation

with st.sidebar:
    method = st.radio("Ranking method", search_engine.METHODS, index=search_engine.METHODS.index("RRF"))
    st.caption(f"Score: {SCORE_HELP[method]}")
    min_ratings = st.number_input(
        "Minimum number of ratings", min_value=0, value=0, step=100,
        help="Hide movies with fewer MovieLens ratings (e.g. 1000 hides obscure titles)",
    )

with st.form("search"):
    query = st.text_input("Search for movies", placeholder="e.g. time travel, funny space adventure, movies about grief")
    submitted = st.form_submit_button("Search")

if not submitted and not query:
    st.info("Type a query and press Search.")
    st.stop()

if not query.strip():
    st.warning("Please enter a search query.")
    st.stop()

try:
    with st.spinner(f"Loading {method} (the first load builds the index and can take a minute)..."):
        load_method(method)
    results = search_engine.search(query, method=method, top_k=TOP_K, min_ratings=min_ratings)
except FileNotFoundError as error:
    st.error(f"Movie data is missing: {error}")
    st.stop()

if not results:
    st.warning(
        "No matching movies found. Keyword methods need at least one non-stop-word "
        "that appears in the data; try other words, the Semantic method, or a lower minimum number of ratings."
    )
    st.stop()

table = pd.DataFrame(results)
columns = ["rank", "title", "year", "genres", "avg_rating", "num_ratings", "score"]
if method == "RRF":
    for name in search_engine.RETRIEVERS:
        # "–" when the movie is not in that method's top results that were fused
        table[f"{name} rank"] = table["ranks"].apply(lambda ranks, name=name: str(ranks.get(name, "–")))
        columns.append(f"{name} rank")
    table["score"] = table["score"].round(5)
else:
    table["score"] = table["score"].round(4)
columns.append("plot")

st.subheader(f"{len(table)} results for “{query.strip()}” ({method})")
st.dataframe(
    table[columns],
    hide_index=True,
    use_container_width=True,
    column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "year": st.column_config.NumberColumn("year", format="%d"),
        "avg_rating": st.column_config.NumberColumn("avg rating", format="%.2f ★", help="Mean MovieLens rating (0.5-5)"),
        "num_ratings": st.column_config.NumberColumn("# ratings", format="%d"),
        "plot": st.column_config.TextColumn("plot", width="large", help="TMDB overview"),
        **{
            f"{name} rank": st.column_config.TextColumn(f"{name} rank", help=f"Position in the {name} list (– = not in its top {search_engine.DEFAULT_RRF_DEPTH})")
            for name in search_engine.RETRIEVERS
        },
    },
)
