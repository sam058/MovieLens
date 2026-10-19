
import pandas as pd
import re
import html
import unicodedata
from pathlib import Path
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Locate the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"

input_path = DATA_DIR / "processed_movies.csv"
output_path = DATA_DIR / "preprocessed_movies.csv"

# 2. Set up the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

# 3. Define the text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""

    # Convert to lowercase and fold accents ("Amélie" -> "amelie") so they are not dropped
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Keep letters and numbers; replace other characters with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize by splitting on whitespace
    tokens = text.split()

    # Remove stop words and stem remaining tokens
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(processed_tokens)


# Placeholder overviews TMDB returns when a movie has no real plot
PLOT_PLACEHOLDERS = {
    "no overview found.",
    "no overview found",
    "we don't have an overview translated in english. help us expand our database by adding one.",
}


def clean_plot(text):
    """Readable plot cleaning: unescape HTML, drop tags, URLs, placeholders and extra whitespace."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in PLOT_PLACEHOLDERS else text


# Guarded so other scripts can import preprocess_text for their queries
if __name__ == "__main__":
    # 4. Load the merged movie dataset
    movies = pd.read_csv(input_path)

    print("Dataset loaded:", len(movies), "movies")

    # 5. Process the searchable text column
    movies["processed_text"] = movies["text"].apply(preprocess_text)

    # 6. Save the result
    movies.to_csv(output_path, index=False)

    print("\nText preprocessing completed successfully!")
    print("Output saved to:", output_path)

    # 7. Compare original and processed text
    print("\nExamples:")
    for _, row in movies.head(5).iterrows():
        print("\nTitle:", row["title"])
        print("Original:", row["text"][:200])
        print("Processed:", row["processed_text"][:200])
