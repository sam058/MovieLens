
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from text_preprocessing import preprocess_text

# 1. Locate the preprocessed dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"
DATA_PATH = DATA_DIR / "preprocessed_movies.csv"

# 2. Load the movie data
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run src/preprocessing.py and "
        "src/text_preprocessing.py first (see README)."
    )

movies = pd.read_csv(DATA_PATH)
movies["processed_text"] = movies["processed_text"].fillna("")

print("Loaded movies:", len(movies))

# 3. Build the BM25 index (the text is already lowercased, stemmed and stop-word free)
bm25 = BM25Okapi(movies["processed_text"].str.split().tolist())

print("BM25 index created!")


# 4. Define the search function
def search_movies(query, top_k=10):
    """Return up to top_k movies as [{"movieId", "title", "score"}], best first.

    score is the BM25 relevance score (unbounded; only the order matters for RRF).
    """
    # Apply the same preprocessing used for movie documents
    tokens = preprocess_text(query).split()

    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:top_k]

    # Exclude movies that share no terms with the query
    return [
        {
            "movieId": int(movies.at[i, "movieId"]),
            "title": movies.at[i, "title"],
            "score": float(scores[i]),
        }
        for i in top_indices
        if scores[i] > 0
    ]


# 5. Run interactive searches
if __name__ == "__main__":
    while True:
        query = input("\nEnter a movie search (or type 'exit'): ")

        if query.lower().strip() == "exit":
            print("Search ended.")
            break

        results = search_movies(query)

        if not results:
            print("No matching movies found.")
            continue

        print("\nSearch Results:")
        print(pd.DataFrame(results).to_string(index=False))
