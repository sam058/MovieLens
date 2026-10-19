
"""Build the test queries and relevance judgments (qrels) from the MovieLens tag genome.

Each topic is a tag-genome tag. The tag genome gives every one of 13,816 movies a
0-1 relevance score for each of 1,128 tags, learned from user tags, ratings and
text reviews (Vig et al., 2012), so every movie in that set is judged for every topic.

  relevant (binary, for P/R/MAP)  : genome relevance >= RELEVANT_THRESHOLD
  graded gain (for nDCG)          : 2 if >= HIGHLY_RELEVANT_THRESHOLD, 1 if relevant, else 0

Each topic gets two queries:
  keyword    - the tag text itself, as a user typing a keyword would
  paraphrase - a natural-language description written without the tag's words,
               fixed before any system was run, so lexical overlap with the
               user tags in the index cannot carry the result

Topics are kept only if they have MIN_RELEVANT..MAX_RELEVANT relevant movies (a rule
fixed up front, not tuned on system results). Outputs (committed, reviewable):
  eval/queries.csv  query_id, topic, query_type, query, n_relevant
  eval/qrels.csv    topic, movieId, genome_relevance, grade
  eval/judged_movies.csv  movieId of every movie with genome scores (the judged pool)
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"
EVAL_DIR = PROJECT_DIR / "eval"

RELEVANT_THRESHOLD = 0.8
HIGHLY_RELEVANT_THRESHOLD = 0.9
MIN_RELEVANT = 10
MAX_RELEVANT = 400

# genome tag -> paraphrase written without the tag's own words
TOPICS = {
    "time travel": "going back in time to change the past",
    "zombies": "undead flesh eaters attacking survivors",
    "vampires": "bloodsucking creatures of the night",
    "dystopia": "oppressive totalitarian society of the future",
    "artificial intelligence": "computers that become self-aware and think for themselves",
    "heist": "a crew planning to rob a bank vault",
    "serial killer": "detective hunting a murderer who kills again and again",
    "post-apocalyptic": "survivors after nuclear war and the collapse of civilization",
    "superhero": "costumed heroes with special powers fighting villains",
    "space travel": "astronauts on a journey between planets",
    "aliens": "extraterrestrial visitors arriving on earth",
    "pirates": "buccaneers sailing the seas in search of treasure",
    "dinosaurs": "prehistoric reptiles brought back to life",
    "martial arts": "kung fu fighters and karate fights",
    "world war ii": "soldiers fighting the nazis in the 1940s",
    "boxing": "a fighter training for the big match in the ring",
    "high school": "teenagers dealing with classes, proms and cliques",
    "road trip": "friends driving across the country together",
    "christmas": "holiday story in december with santa claus and presents",
    "mafia": "an organized crime family and its godfather",
    "cyberpunk": "hackers in a neon high-tech future ruled by corporations",
    "courtroom drama": "lawyers arguing a case before a judge and jury",
    "submarine": "naval crew trapped in a vessel under the ocean",
    "robots": "mechanical androids that act like humans",
    "ghosts": "haunted house with spirits of the dead",
    "werewolves": "a man who turns into a wolf under the full moon",
    "spies": "secret agents on an espionage mission",
    "prison": "inmates serving time behind bars trying to escape",
    "wedding": "bride and groom getting married",
    "boarding school": "students living away from home at an elite academy",
}


def main():
    genome_tags = pd.read_csv(DATA_DIR / "genome-tags.csv")
    scores = pd.read_csv(DATA_DIR / "genome-scores.csv")

    tag_ids = dict(zip(genome_tags["tag"], genome_tags["tagId"]))
    missing = [topic for topic in TOPICS if topic not in tag_ids]
    if missing:
        raise SystemExit(f"Not genome tags: {missing}")

    queries, qrels, skipped = [], [], []
    for topic, paraphrase in TOPICS.items():
        topic_scores = scores[scores["tagId"] == tag_ids[topic]]
        relevant = topic_scores[topic_scores["relevance"] >= RELEVANT_THRESHOLD]
        if not MIN_RELEVANT <= len(relevant) <= MAX_RELEVANT:
            skipped.append((topic, len(relevant)))
            continue

        qrels.append(pd.DataFrame({
            "topic": topic,
            "movieId": relevant["movieId"].to_numpy(),
            "genome_relevance": relevant["relevance"].round(4).to_numpy(),
            "grade": (relevant["relevance"] >= HIGHLY_RELEVANT_THRESHOLD).astype(int).to_numpy() + 1,
        }))
        for query_type, text in (("keyword", topic), ("paraphrase", paraphrase)):
            queries.append({
                "query_id": f"q{len(queries) + 1:02d}",
                "topic": topic,
                "query_type": query_type,
                "query": text,
                "n_relevant": len(relevant),
            })

    EVAL_DIR.mkdir(exist_ok=True)
    pd.DataFrame(queries).to_csv(EVAL_DIR / "queries.csv", index=False)
    pd.concat(qrels).sort_values(["topic", "movieId"]).to_csv(EVAL_DIR / "qrels.csv", index=False)
    pd.DataFrame({"movieId": sorted(scores["movieId"].unique())}).to_csv(
        EVAL_DIR / "judged_movies.csv", index=False
    )

    print(f"Kept {len(queries) // 2} topics ({len(queries)} queries).")
    for topic, count in skipped:
        print(f"Skipped {topic!r}: {count} relevant movies (allowed {MIN_RELEVANT}-{MAX_RELEVANT})")
    print("Judged movies:", scores["movieId"].nunique())
    print("Saved to", EVAL_DIR)


if __name__ == "__main__":
    main()
