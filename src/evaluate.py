
"""Evaluate TF-IDF, BM25, Semantic and RRF on eval/queries.csv + eval/qrels.csv.

Run src/build_relevance.py first. Writes to results/:
  per_query_metrics.csv   every metric for every query x method
  comparison.csv / .md    mean metrics per method (the comparison table)
  pr_curve.png / .csv     11-point interpolated precision-recall curves

Rankings are restricted to the 13,816 movies with tag-genome scores, so every
retrieved movie has a relevance judgment.

Usage: python src/evaluate.py [--rrf-k 60] [--depth 1000]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import metrics  # noqa: E402
from rrf import DEFAULT_K, reciprocal_rank_fusion  # noqa: E402
from search_engine import METHODS, RETRIEVERS, get_retriever  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_DIR / "eval"
RESULTS_DIR = PROJECT_DIR / "results"

CUTOFFS = (5, 10)
METRIC_COLUMNS = ["P@5", "P@10", "R@5", "R@10", "MAP", "nDCG@5", "nDCG@10"]

# Validated categorical palette, light mode, slots 1-4 (dataviz reference palette)
COLORS = {"TF-IDF": "#2a78d6", "BM25": "#eb6834", "Semantic": "#1baf7a", "RRF": "#eda100"}
MARKERS = {"TF-IDF": "o", "BM25": "s", "Semantic": "^", "RRF": "D"}
LINESTYLES = {"TF-IDF": "-", "BM25": "--", "Semantic": "-.", "RRF": "-"}


def load_eval_data():
    queries = pd.read_csv(EVAL_DIR / "queries.csv")
    qrels = pd.read_csv(EVAL_DIR / "qrels.csv")
    judged = set(pd.read_csv(EVAL_DIR / "judged_movies.csv")["movieId"])
    relevant = qrels.groupby("topic")["movieId"].apply(set).to_dict()
    gains = {
        topic: dict(zip(group["movieId"], group["grade"]))
        for topic, group in qrels.groupby("topic")
    }
    return queries, relevant, gains, judged


def rankings(query, judged, depth, rrf_k):
    """Ranked movieId lists per method for one query, restricted to judged movies."""
    lists = {}
    for name in RETRIEVERS:
        results = get_retriever(name)(query, top_k=None)
        lists[name] = [r for r in results if r["movieId"] in judged][:depth]
    fused = reciprocal_rank_fusion(lists, k=rrf_k, top_k=depth)
    ranked = {name: [r["movieId"] for r in results] for name, results in lists.items()}
    ranked["RRF"] = [r["movieId"] for r in fused]
    return ranked


def score_ranking(ranked, relevant, gains):
    row = {}
    for k in CUTOFFS:
        row[f"P@{k}"] = metrics.precision_at_k(ranked, relevant, k)
        row[f"R@{k}"] = metrics.recall_at_k(ranked, relevant, k)
        row[f"nDCG@{k}"] = metrics.ndcg_at_k(ranked, gains, k)
    row["MAP"] = metrics.average_precision(ranked, relevant)
    return row


def plot_pr_curves(curves, path):
    """curves: {method: mean interpolated precision at each recall level}"""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for method in METHODS:
        ax.plot(
            metrics.RECALL_LEVELS, curves[method], label=method, color=COLORS[method],
            marker=MARKERS[method], linestyle=LINESTYLES[method], linewidth=2, markersize=6,
        )
    ax.set_title("Precision-recall (11-point interpolated, mean over queries)", fontsize=11)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b8b7b1")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_markdown(table):
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(table[c]) else "---" for c in table.columns) + "|"
    rows = [
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rrf-k", type=float, default=DEFAULT_K)
    parser.add_argument("--depth", type=int, default=1000, help="ranking depth scored per query")
    args = parser.parse_args()

    queries, relevant, gains, judged = load_eval_data()
    print(f"{len(queries)} queries over {queries['topic'].nunique()} topics")

    rows, curves = [], {method: [] for method in METHODS}
    for q in queries.itertuples():
        ranked = rankings(q.query, judged, args.depth, args.rrf_k)
        for method in METHODS:
            rows.append({"query_id": q.query_id, "query": q.query, "method": method,
                         **score_ranking(ranked[method], relevant[q.topic], gains[q.topic])})
            curves[method].append(metrics.interpolated_precision(ranked[method], relevant[q.topic]))

    per_query = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    per_query.to_csv(RESULTS_DIR / "per_query_metrics.csv", index=False)

    comparison = per_query.groupby("method", sort=False)[METRIC_COLUMNS].mean().reindex(METHODS).reset_index()
    comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False, float_format="%.4f")

    mean_curves = {method: np.mean(c, axis=0) for method, c in curves.items()}
    plot_pr_curves(mean_curves, RESULTS_DIR / "pr_curve.png")
    pd.DataFrame({"recall": metrics.RECALL_LEVELS, **mean_curves}).to_csv(
        RESULTS_DIR / "pr_curve.csv", index=False, float_format="%.4f"
    )

    report = (
        f"# Evaluation results\n\n{len(queries)} queries ({queries['topic'].nunique()} tag-genome topics), "
        f"RRF k={args.rrf_k:g}, ranking depth {args.depth}. Generated by `python src/evaluate.py`.\n\n"
        + to_markdown(comparison) + "\n"
    )
    (RESULTS_DIR / "comparison.md").write_text(report)

    print("\n" + report)
    print("Saved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
