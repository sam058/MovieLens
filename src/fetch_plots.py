
"""Fetch plot descriptions from TMDB for every MovieLens movie (MovieLens has none).

links.csv maps each movieId to a TMDB id; this script requests
https://api.themoviedb.org/3/movie/{tmdbId} and keeps the English "overview".

Credentials (from the environment or a git-ignored .env in the project root), either:
  TMDB_READ_ACCESS_TOKEN=...   (API Read Access Token, sent as a Bearer header; preferred)
  TMDB_API_KEY=...             (v3 API key)
Get one free at https://www.themoviedb.org/settings/api.

Results are cached in data/ml-25m/plots.csv (movieId, tmdbId, status, plot). The run is
resumable: rows already fetched ("ok" / "not_found") are skipped and errors are retried.

Usage: python src/fetch_plots.py [--limit N] [--workers 8]
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"
LINKS_PATH = DATA_DIR / "links.csv"
PLOTS_PATH = DATA_DIR / "plots.csv"
ENV_PATH = PROJECT_DIR / ".env"

API_URL = "https://api.themoviedb.org/3/movie/{}"
FIELDS = ["movieId", "tmdbId", "status", "plot"]
DONE_STATUSES = {"ok", "not_found"}
MAX_RETRIES = 5


def load_credentials():
    """Return request headers and query parameters for TMDB authentication."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    token = os.environ.get("TMDB_READ_ACCESS_TOKEN")
    api_key = os.environ.get("TMDB_API_KEY")
    if token:
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}, {}
    if api_key:
        return {"Accept": "application/json"}, {"api_key": api_key}
    raise SystemExit(
        "No TMDB credentials. Put TMDB_READ_ACCESS_TOKEN=... (or TMDB_API_KEY=...) in .env "
        "in the project root; see the README."
    )


def fetch_overview(tmdb_id, headers, params, opener=urllib.request.urlopen):
    """Return (status, overview) for one TMDB id; status is ok, not_found or error."""
    query = urllib.parse.urlencode({"language": "en-US", **params})
    request = urllib.request.Request(f"{API_URL.format(tmdb_id)}?{query}", headers=headers)
    for attempt in range(MAX_RETRIES):
        try:
            with opener(request, timeout=20) as response:
                # Collapse whitespace: some overviews contain stray carriage returns
                return "ok", " ".join((json.load(response).get("overview") or "").split())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "not_found", ""
            if error.code == 401:
                raise SystemExit("TMDB rejected the credentials (HTTP 401).") from error
            if error.code == 429 or error.code >= 500:
                time.sleep(float(error.headers.get("Retry-After") or 2**attempt))
                continue
            return "error", ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2**attempt)
    return "error", ""


def load_plots():
    """Read the plot cache, skipping malformed rows (those movies are fetched again)."""
    if not PLOTS_PATH.exists():
        return pd.DataFrame(columns=FIELDS)
    with open(PLOTS_PATH, newline="") as file:
        rows = [row for row in csv.reader(file) if len(row) == len(FIELDS)]
    plots = pd.DataFrame(rows[1:], columns=FIELDS)
    plots["movieId"] = pd.to_numeric(plots["movieId"], errors="coerce")
    plots = plots.dropna(subset=["movieId"]).astype({"movieId": int})
    plots["plot"] = plots["plot"].str.split().str.join(" ")
    return plots.drop_duplicates("movieId", keep="last")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="fetch at most N movies (for a quick test)")
    parser.add_argument("--workers", type=int, default=8, help="parallel requests (TMDB allows ~40/s)")
    args = parser.parse_args()

    headers, params = load_credentials()
    links = pd.read_csv(LINKS_PATH).dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    cache = load_plots()
    done = set(cache.loc[cache["status"].isin(DONE_STATUSES), "movieId"].astype(int))
    todo = links[~links["movieId"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(links)} movies with a TMDB id, {len(done)} already fetched, {len(todo)} to fetch")

    # Keep only finished rows, then append new results as they arrive (resumable)
    cache[cache["status"].isin(DONE_STATUSES)].to_csv(PLOTS_PATH, index=False, lineterminator="\n")
    counts = {"ok": 0, "not_found": 0, "error": 0}
    start = time.time()
    with open(PLOTS_PATH, "a", newline="") as file, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.writer(file, lineterminator="\n")
        futures = {
            pool.submit(fetch_overview, row.tmdbId, headers, params): row
            for row in todo.itertuples()
        }
        for i, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            status, plot = future.result()
            counts[status] += 1
            writer.writerow([row.movieId, row.tmdbId, status, plot])
            if i % 1000 == 0:
                file.flush()
                print(f"  {i}/{len(todo)} ({i / (time.time() - start):.0f}/s) {counts}")

    print("Done:", counts, "->", PLOTS_PATH)
    if counts["error"]:
        print("Run the script again to retry the errors.")


if __name__ == "__main__":
    main()
