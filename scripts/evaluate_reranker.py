#!/usr/bin/env python3
"""A2 Q5 -- extended evaluation of the trained re-ranker (scripts/train_reranker.py)
on val: AUC/MRR/nDCG@5/nDCG@10/diversity/novelty/coverage, sliced by
cold-start-vs-warm (history length) AND head-vs-tail (clicked-item
popularity), bootstrap 95% CIs on every metric, and -- per Q3's "claimed
gains must ship a paired bootstrap 95% CI that excludes zero" -- a PAIRED
bootstrap CI on the re-ranker's improvement over the fusion-only baseline,
not just two independent CIs side by side.

Loads the ALREADY-TRAINED model (data/models/<dataset>/reranker.pkl) --
does not retrain. Mirrors scripts/evaluate_ranking.py's metric conventions
(per-impression, group-averaged; diversity always computed in the semantic
embedding space regardless of which method produced the ranking) plus
scripts/train_reranker.py's data-loading path (retrieval/reranker_data.py),
so numbers here are directly comparable to both.

Slicing definitions:
  - cold/warm: this impression's own history_article_ids length < / >=
    --cold_start_threshold (same convention as scripts/evaluate_ranking.py).
  - head/tail: whether any of THIS impression's actually-clicked
    candidate(s) is a "head" (popular) article -- train-split exposure
    count at or above the --head_percentile-th percentile across the whole
    catalog, else "tail". An impression with zero clicks has no clicked
    item to classify by and is excluded from the head/tail slice (it's
    already excluded from AUC/MRR/nDCG for the same reason -- those are
    undefined without a positive label).

    python scripts/train_reranker.py --dataset mind     # once, if not already trained
    python scripts/evaluate_reranker.py --dataset mind
    python scripts/evaluate_reranker.py --dataset ebnerd

Writes data/reports/<dataset>_reranker_extended_eval.json.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.beyond_accuracy import intra_list_diversity, novelty_score, train_popularity_prob
from retrieval.bootstrap import bootstrap_ci_coverage, bootstrap_ci_mean, bootstrap_ci_paired_delta
from retrieval.build_indices import load_indices
from retrieval.ranking_metrics import auc_score, mrr_score, ndcg_at_k
from retrieval.reranker_data import FEATURE_COLUMNS, load_split_features
from retrieval.text_utils import article_text

METHODS = ["before_fusion", "after_reranker"]
SCORE_COL = {"before_fusion": "fusion_score", "after_reranker": "reranker_score"}
SCALAR_METRICS = ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]


def classify_slices(val_df, cold_start_threshold, head_threshold):
    """One row per impression: (impression_id, slice_cold_warm,
    slice_head_tail). head/tail uses the MAX article_train_impressions
    among that impression's actually-clicked candidates -- "none" if it
    has no click at all (excluded from the head/tail slice, not forced
    into either bucket)."""
    def _classify(g):
        hist_len = g["history_length"].iloc[0]
        clicked = g.loc[g["label"] == 1, "article_train_impressions"]
        if len(clicked) == 0:
            head_tail = "none"
        else:
            head_tail = "head" if clicked.max() >= head_threshold else "tail"
        return pd.Series({
            "cold_warm": "cold" if hist_len < cold_start_threshold else "warm",
            "head_tail": head_tail,
        })
    return val_df.groupby("impression_id", sort=False, observed=True).apply(_classify, include_groups=False)


def evaluate_full(val_df, semantic, pop_lookup, k_beyond):
    """One pass over all impressions, computing every scalar metric for
    BOTH before/after score columns at once (rather than two separate
    passes) -- returns per-impression arrays (for slicing/bootstrap) plus
    the flattened (impression_owner, item_row) arrays coverage needs, all
    keyed by method."""
    per_impression = {m: {k: [] for k in SCALAR_METRICS} for m in METHODS}
    item_owner = {m: [] for m in METHODS}
    item_row = {m: [] for m in METHODS}
    impression_ids = []

    for i, (imp_id, g) in enumerate(val_df.groupby("impression_id", sort=False, observed=True)):
        impression_ids.append(imp_id)
        labels = g["label"].to_numpy()
        cand_ids = g["candidate_article_id"].tolist()
        for method in METHODS:
            scores = g[SCORE_COL[method]].to_numpy()
            per_impression[method]["auc"].append(auc_score(labels, scores))
            per_impression[method]["mrr"].append(mrr_score(labels, scores))
            per_impression[method]["ndcg5"].append(ndcg_at_k(labels, scores, 5))
            per_impression[method]["ndcg10"].append(ndcg_at_k(labels, scores, 10))

            order = np.argsort(-scores, kind="stable")[:k_beyond]
            topk_ids = [cand_ids[j] for j in order]
            topk_rows = [semantic.id_to_row.get(a) for a in topk_ids]
            topk_rows = [r for r in topk_rows if r is not None]
            topk_embs = semantic.embeddings[topk_rows] if topk_rows else []

            per_impression[method]["diversity"].append(intra_list_diversity(topk_embs))
            per_impression[method]["novelty"].append(novelty_score(topk_ids, pop_lookup))

            item_owner[method].extend([i] * len(topk_rows))
            item_row[method].extend(topk_rows)

    return impression_ids, per_impression, item_owner, item_row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--k_beyond", type=int, default=10, help="top-K cutoff for diversity/novelty/coverage")
    ap.add_argument("--cold_start_threshold", type=int, default=5)
    ap.add_argument("--head_percentile", type=float, default=80.0,
                     help="an article at/above this train-exposure percentile counts as a 'head' item")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="cap on val impressions, for a quick test run")
    args = ap.parse_args()

    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)
    # Load order matters here, concretely (not superstition): unpickling
    # LightGBM's reranker.pkl AFTER semantic.pkl (an SBERTIndex, which
    # rebuilds a FAISS index on unpickle -- see retrieval/sbert.py's
    # __setstate__) segfaults every time on this machine, while loading
    # the LightGBM model FIRST and the FAISS-backed index second does not
    # -- confirmed by isolating it to exactly this pair and this order via
    # a standalone repro before touching this file. A native
    # library-initialization conflict between FAISS and LightGBM (most
    # likely both trying to set up their own OpenMP/threading runtime),
    # not something fixable from Python beyond not triggering the bad
    # order.
    with open(os.path.join(model_dir, "reranker.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "reranker_config.json")) as f:
        rr_config = json.load(f)
    bm25, semantic, index_config = load_indices(model_dir)
    feature_cols = rr_config["feature_columns"]
    recent_n, recency_decay, fusion_alpha = rr_config["recent_n"], rr_config["recency_decay"], rr_config["fusion_alpha"]
    print(f"Dataset={args.dataset}, loaded reranker trained with {len(feature_cols)} features, "
          f"recent_n={recent_n}, recency_decay={recency_decay}, fusion_alpha={fusion_alpha}")

    articles = pd.read_parquet(os.path.join(args.data_dir, "processed", args.dataset, "articles.parquet"))
    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))
    entity_lookup = dict(zip(articles["article_id"], articles["entities"]))

    print("Loading val split (reusing cached Assignment-1 retrieval scores if present) ...")
    val_df = load_split_features(args.dataset, "val", args.data_dir, bm25, semantic, text_lookup,
                                   entity_lookup, recent_n, recency_decay, fusion_alpha, args.limit,
                                   feature_cols=feature_cols)
    print(f"  val: {len(val_df):,} rows / {val_df['impression_id'].nunique():,} impressions")
    val_df["reranker_score"] = model.predict(val_df[feature_cols])

    article_features = pd.read_parquet(
        os.path.join(args.data_dir, "feature_store", args.dataset, "article_features.parquet"))
    pop_lookup = train_popularity_prob(article_features)
    head_threshold = article_features["train_impressions"].quantile(args.head_percentile / 100.0)
    print(f"  head/tail threshold: train_impressions >= {head_threshold:.1f} "
          f"(the {args.head_percentile:.0f}th percentile across {len(article_features):,} articles)")

    print("Classifying cold/warm and head/tail slices ...")
    slices = classify_slices(val_df, args.cold_start_threshold, head_threshold)

    print(f"Scoring all impressions (before=fusion_score, after=reranker_score, k_beyond={args.k_beyond}) ...")
    t0 = time.time()
    impression_ids, per_impression, item_owner, item_row = evaluate_full(val_df, semantic, pop_lookup, args.k_beyond)
    n_impressions = len(impression_ids)
    print(f"  scored {n_impressions:,} impressions in {time.time()-t0:.1f}s")

    slice_arr = slices.reindex(impression_ids)
    cold_warm = slice_arr["cold_warm"].to_numpy()
    head_tail = slice_arr["head_tail"].to_numpy()
    slice_masks = {
        "overall": np.ones(n_impressions, dtype=bool),
        "cold": cold_warm == "cold",
        "warm": cold_warm == "warm",
        "head": head_tail == "head",
        "tail": head_tail == "tail",
    }
    print(f"  slice counts: cold={slice_masks['cold'].sum():,}, warm={slice_masks['warm'].sum():,}, "
          f"head={slice_masks['head'].sum():,}, tail={slice_masks['tail'].sum():,} "
          f"(head+tail < overall: {(head_tail == 'none').sum():,} impressions have no click to classify by)")

    report = {
        "dataset": args.dataset, "n_val_impressions": n_impressions,
        "k_beyond": args.k_beyond, "cold_start_threshold": args.cold_start_threshold,
        "head_percentile": args.head_percentile, "head_threshold_train_impressions": float(head_threshold),
        "n_boot": args.n_boot, "reranker_config": rr_config,
        "slice_counts": {k: int(v.sum()) for k, v in slice_masks.items()},
        "results": {}, "paired_delta_vs_fusion": {},
    }

    for method in METHODS:
        report["results"][method] = {}
        arrs = {k: np.array(v) for k, v in per_impression[method].items()}
        io_arr, ir_arr = np.array(item_owner[method]), np.array(item_row[method])
        for slice_name, mask in slice_masks.items():
            slice_report = {m: bootstrap_ci_mean(arrs[m][mask], n_boot=args.n_boot) for m in SCALAR_METRICS}
            global_idx = np.where(mask)[0]
            local_id = -np.ones(n_impressions, dtype=int)
            local_id[global_idx] = np.arange(len(global_idx))
            keep = local_id[io_arr] >= 0
            slice_report["coverage"] = bootstrap_ci_coverage(
                local_id[io_arr[keep]], ir_arr[keep], len(global_idx), index_config["n_docs"], n_boot=args.n_boot)
            report["results"][method][slice_name] = slice_report

    # -- the actual significance test: paired delta, not two independent CIs --
    before_arrs = {k: np.array(v) for k, v in per_impression["before_fusion"].items()}
    after_arrs = {k: np.array(v) for k, v in per_impression["after_reranker"].items()}
    for slice_name, mask in slice_masks.items():
        report["paired_delta_vs_fusion"][slice_name] = {
            m: bootstrap_ci_paired_delta(before_arrs[m][mask], after_arrs[m][mask], n_boot=args.n_boot)
            for m in SCALAR_METRICS
        }

    print(f"\n=== Extended evaluation: {args.dataset} (before=fusion score, after=reranker) ===")
    for slice_name in slice_masks:
        print(f"\n-- {slice_name} ({report['slice_counts'][slice_name]:,} impressions) --")
        header = f"{'metric':<10}{'before':>10}{'after':>10}{'delta':>10}{'CI':>22}{'excl.0':>8}"
        print(header)
        for m in SCALAR_METRICS:
            b = report["results"]["before_fusion"][slice_name][m]["mean"]
            a = report["results"]["after_reranker"][slice_name][m]["mean"]
            d = report["paired_delta_vs_fusion"][slice_name][m]
            ci_str = f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"
            print(f"{m:<10}{b:>10.4f}{a:>10.4f}{d['delta']:>+10.4f}{ci_str:>22}{str(d['excludes_zero']):>8}")
        cb = report["results"]["before_fusion"][slice_name]["coverage"]["point_estimate"]
        ca = report["results"]["after_reranker"][slice_name]["coverage"]["point_estimate"]
        print(f"{'coverage':<10}{cb:>10.4f}{ca:>10.4f}{ca-cb:>+10.4f}"
              f"{'(coverage has no paired-delta CI; see docstring)':>30}")

    report_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_reranker_extended_eval.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
