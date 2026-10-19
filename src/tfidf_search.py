
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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

# Remove empty documents
movies = movies[movies["processed_text"].str.strip() != ""].reset_index(drop=True)

print("Loaded movies:", len(movies))

# 3. Create the TF-IDF model
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(movies["processed_text"])

print("TF-IDF model created!")
print("Matrix shape:", tfidf_matrix.shape)


# 4. Define the search function
def search_movies(query, top_k=10):
    """Return up to top_k movies as [{"movieId", "title", "score"}], best first.

    score is the cosine similarity of the TF-IDF vectors (0 to 1).
    Returns [] for an empty query or when no movie shares a term with it.
    """
    # Apply the same preprocessing used for movie documents
    processed_query = preprocess_text(query) if isinstance(query, str) else ""

    if not processed_query:
        return []

    # Convert query to a TF-IDF vector
    query_vector = vectorizer.transform([processed_query])

    # Rows are L2-normalized, so the dot product is the cosine similarity
    scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    # Retrieve the top results
    top_indices = scores.argsort()[::-1][:top_k]

    # Exclude zero-score matches
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
