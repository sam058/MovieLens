
import pandas as pd
from pathlib import Path
from fetch_plots import load_plots
from text_preprocessing import clean_plot

# 1. Locate the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"

# 2. Load the CSV files
movies = pd.read_csv(DATA_DIR / "movies.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")

print("Movies loaded:", len(movies))
print("Tags loaded:", len(tags))

# 2b. Plot descriptions fetched from TMDB by src/fetch_plots.py (MovieLens has none)
PLOTS_PATH = DATA_DIR / "plots.csv"
if PLOTS_PATH.exists():
    plots = load_plots()
    plots = plots[plots["status"] == "ok"]
    plots = plots.assign(plot=plots["plot"].map(clean_plot))[["movieId", "plot"]]
else:
    print("No plots.csv found - run src/fetch_plots.py to add plot descriptions")
    plots = pd.DataFrame(columns=["movieId", "plot"])

print("Movies with a plot:", int((plots["plot"] != "").sum()))

# 3. Remove rows where the tag is missing
tags = tags.dropna(subset=["tag"])

# 4. Combine all tags belonging to each movie
movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(lambda values: " ".join(values.astype(str)))
    .reset_index(name="tags")
)

print("Movies with at least one tag:", len(movie_tags))

# 4b. Distinct tags per movie, most-applied first, for the sentence embedding model.
# all-MiniLM-L6-v2 reads only the first 256 word pieces, and popular movies have
# thousands of repeated tags, so the full "tags" text would be cut off arbitrarily.
unique_tags = (
    tags.assign(tag=tags["tag"].astype(str).str.lower().str.strip())
    .groupby(["movieId", "tag"])
    .size()
    .reset_index(name="count")
    .sort_values(["movieId", "count", "tag"], ascending=[True, False, True])
    .groupby("movieId")["tag"]
    .apply(", ".join)
    .reset_index(name="unique_tags")
)

# 5. Merge the tags into the movie dataset
combined = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)

combined = combined.merge(unique_tags, on="movieId", how="left")
combined = combined.merge(plots, on="movieId", how="left")

# 6. Replace missing tags and plots with empty strings
combined["tags"] = combined["tags"].fillna("")
combined["unique_tags"] = combined["unique_tags"].fillna("")
combined["plot"] = combined["plot"].fillna("")

# 7. Convert genres into space-separated words
combined["genres_text"] = (
    combined["genres"]
    .fillna("")
    .str.replace("|", " ", regex=False)
    .str.replace("(no genres listed)", "", regex=False)
)

# 8. Create the searchable text for each movie
combined["text"] = (
    combined["title"].fillna("")
    + " "
    + combined["genres_text"]
    + " "
    + combined["plot"]
    + " "
    + combined["tags"]
).str.strip()

# 8b. Readable text for semantic search: plot, then each tag once, most-applied first
genres_part = combined["genres_text"].str.split().str.join(" ")
plot_part = combined["plot"].str.rstrip(".")
combined["semantic_text"] = (
    combined["title"].fillna("")
    + (". Genres: " + genres_part).where(genres_part != "", "")
    + (". Plot: " + plot_part).where(plot_part != "", "")
    + (". Tags: " + combined["unique_tags"]).where(combined["unique_tags"] != "", "")
).str.strip()

# 9. Save the combined dataset
OUTPUT_DIR = DATA_DIR
output_path = OUTPUT_DIR / "processed_movies.csv"

combined.to_csv(output_path, index=False)

print("\nMerging completed successfully!")
print("Output saved to:", output_path)

# 9b. Metadata index: year, genres, rating statistics and external ids per movie
ratings = pd.read_csv(DATA_DIR / "ratings.csv", usecols=["movieId", "rating"])
rating_stats = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", num_ratings="size").reset_index()
links = pd.read_csv(DATA_DIR / "links.csv")

metadata = (
    combined[["movieId", "title", "genres", "plot"]]
    .merge(rating_stats, on="movieId", how="left")
    .merge(links, on="movieId", how="left")
)
# Year is the last "(YYYY)" in the title; some titles have none
metadata["year"] = metadata["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("Int64")
metadata["avg_rating"] = metadata["avg_rating"].round(2)
metadata["num_ratings"] = metadata["num_ratings"].fillna(0).astype(int)
metadata["has_plot"] = metadata["plot"] != ""
metadata["tmdbId"] = metadata["tmdbId"].astype("Int64")
metadata = metadata[["movieId", "title", "year", "genres", "avg_rating", "num_ratings", "has_plot", "imdbId", "tmdbId", "plot"]]

metadata_path = OUTPUT_DIR / "movie_metadata.csv"
metadata.to_csv(metadata_path, index=False)
print("Metadata saved to:", metadata_path)
print("Movies with a year:", int(metadata["year"].notna().sum()), "| with ratings:", int((metadata["num_ratings"] > 0).sum()))

# 10. Display a few examples
print("\nSample combined movies:")
print(combined[["title", "genres", "plot", "tags"]].head(5).to_string(index=False, max_colwidth=60))
