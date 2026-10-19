
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. Locate the dataset and the saved embeddings
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"
DATA_PATH = DATA_DIR / "processed_movies.csv"
EMBEDDINGS_PATH = PROJECT_DIR / "models" / "movie_embeddings.npy"

# 2. Load the movie data (readable text keeps the meaning the model needs)
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run src/preprocessing.py first (see README)."
    )

movies = pd.read_csv(DATA_PATH)

# Prefer the de-duplicated tag text (fits the model's 256-token window);
# older CSVs without it fall back to the full text
TEXT_COLUMN = "semantic_text" if "semantic_text" in movies.columns else "text"
movies[TEXT_COLUMN] = movies[TEXT_COLUMN].fillna("")

print("Loaded movies:", len(movies))

# 3. Load the sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Load the saved movie embeddings, or create them once and save them
def load_embeddings():
    # Reuse the saved file only if it is newer than the CSV and has one row per movie
    if (
        EMBEDDINGS_PATH.exists()
        and EMBEDDINGS_PATH.stat().st_mtime > DATA_PATH.stat().st_mtime
    ):
        embeddings = np.load(EMBEDDINGS_PATH)
        if len(embeddings) == len(movies):
            print("Loaded saved embeddings:", embeddings.shape)
            return embeddings

    print("Creating movie embeddings (one-time, takes several minutes on CPU)...")

    # Normalized vectors make the dot product equal to cosine similarity
    embeddings = model.encode(
        movies[TEXT_COLUMN].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    EMBEDDINGS_PATH.parent.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print("Embeddings saved to:", EMBEDDINGS_PATH)

    return embeddings


embeddings = load_embeddings()


# 5. Define the search function
def search_movies(query, top_k=10):
    """Return up to top_k movies as [{"movieId", "title", "score"}], best first.

    score is the cosine similarity between query and movie text (-1 to 1).
    """
    if not isinstance(query, str) or not query.strip():
        return []

    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_embedding

    top_indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "movieId": int(movies.at[i, "movieId"]),
            "title": movies.at[i, "title"],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


# 6. Run interactive searches
if __name__ == "__main__":
    while True:
        query = input("\nEnter a movie search (or type 'exit'): ")

        if query.lower().strip() == "exit":
            print("Search ended.")
            break

        results = search_movies(query)

        if not results:
            print("Please enter a search query.")
            continue

        print("\nSearch Results:")
        print(pd.DataFrame(results).to_string(index=False))
