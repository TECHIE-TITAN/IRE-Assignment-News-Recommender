#!/usr/bin/env python3
"""Q4: offline evaluation harness.

Reranks each MINDsmall_dev ("val" split) impression's own candidate list
with the BM25 and LSA scorers (the same `score_candidates` used for Q5
prediction generation, not the full-corpus retrieval from
scripts/evaluate_retrieval.py -- that's a candidate-generation diagnostic,
this is ranking-quality evaluation) and reports:

  - accuracy: AUC, MRR, nDCG@5, nDCG@10 (group-averaged per impression,
    MIND's own official-evaluate.py convention)
  - beyond-accuracy: intra-list diversity, novelty, catalog coverage
  - a cold-start (few history clicks) vs warm slice
  - bootstrap 95% CIs for every metric above, for both methods

    python scripts/evaluate_ranking.py

Reads data/processed/mind/{articles,interactions}.parquet and
data/feature_store/mind/article_features.parquet (all built by
build_pipeline.py). Writes data/reports/mind_ranking_eval.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.beyond_accuracy import intra_list_diversity, novelty_score, train_popularity_prob
from retrieval.bootstrap import bootstrap_ci_coverage, bootstrap_ci_mean
from retrieval.build_indices import fit_indices
from retrieval.lsa import mean_pool_user_vector
from retrieval.ranking_metrics import auc_score, mrr_score, ndcg_at_k
from retrieval.text_utils import tokenize

METHODS = ["bm25", "lsa"]
SCALAR_METRICS = ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--recent_n", type=int, default=20, help="history articles used to build the query")
    ap.add_argument("--lsa_components", type=int, default=128)
    ap.add_argument("--k_beyond", type=int, default=10, help="top-K cutoff for diversity/novelty/coverage")
    ap.add_argument("--cold_start_threshold", type=int, default=5,
                     help="impressions with fewer than this many history articles are the 'cold' slice")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed", "mind")
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    val = interactions[interactions["split"] == "val"].reset_index(drop=True)
    article_features = pd.read_parquet(os.path.join(args.data_dir, "feature_store", "mind", "article_features.parquet"))
    print(f"Loaded {len(articles):,} articles, {len(val):,} val impressions")

    title_lookup = dict(zip(articles["article_id"], articles["title"]))
    pop_lookup = train_popularity_prob(article_features)

    bm25, lsa, doc_ids = fit_indices(articles, lsa_components=args.lsa_components)
    n_docs = len(doc_ids)

    # -- per-impression fields as plain lists (fast to iterate) ---------------
    hist_list = val["history_article_ids"].tolist()
    cand_list = val["candidate_article_ids"].tolist()
    label_list = val["labels"].tolist()
    n_val = len(val)

    cold_threshold = args.cold_start_threshold
    slice_of = np.array(["cold" if len(h) < cold_threshold else "warm" for h in hist_list])
    print(f"Slicing @ history_length < {cold_threshold}: "
          f"{(slice_of == 'cold').sum():,} cold, {(slice_of == 'warm').sum():,} warm")

    metrics = {m: {s: [] for s in SCALAR_METRICS} for m in METHODS}
    item_owner = {m: [] for m in METHODS}
    item_row = {m: [] for m in METHODS}

    t_start = time.time()
    for i in range(n_val):
        hist = hist_list[i]
        cand_ids = cand_list[i]
        labels = np.array(label_list[i])
        recent = hist[-args.recent_n:] if len(hist) else []

        q_tokens = tokenize(" ".join(title_lookup.get(a, "") for a in recent if title_lookup.get(a)))
        bm25_scores = bm25.score_candidates(q_tokens, cand_ids)

        embs = [lsa.get_embedding(a) for a in recent]
        user_vec = mean_pool_user_vector(embs)
        lsa_scores = lsa.score_candidates(user_vec, cand_ids)

        for method, scores in [("bm25", bm25_scores), ("lsa", lsa_scores)]:
            metrics[method]["auc"].append(auc_score(labels, scores))
            metrics[method]["mrr"].append(mrr_score(labels, scores))
            metrics[method]["ndcg5"].append(ndcg_at_k(labels, scores, 5))
            metrics[method]["ndcg10"].append(ndcg_at_k(labels, scores, 10))

            order = np.argsort(-scores, kind="stable")[:args.k_beyond]
            topk_ids = [cand_ids[j] for j in order]
            topk_rows = [lsa.id_to_row.get(a) for a in topk_ids]
            topk_rows = [r for r in topk_rows if r is not None]
            topk_embs = [lsa.embeddings[r] for r in topk_rows]

            metrics[method]["diversity"].append(intra_list_diversity(topk_embs))
            metrics[method]["novelty"].append(novelty_score(topk_ids, pop_lookup))

            item_owner[method].extend([i] * len(topk_rows))
            item_row[method].extend(topk_rows)

        if i % 10_000 == 0:
            print(f"  {i:,}/{n_val:,} impressions scored ({time.time()-t_start:.1f}s elapsed)")

    print(f"Scored all {n_val:,} impressions in {time.time()-t_start:.1f}s. Computing bootstrap CIs "
          f"(n_boot={args.n_boot}) ...")

    report = {
        "n_val_impressions": int(n_val),
        "cold_start_threshold": cold_threshold,
        "k_beyond": args.k_beyond,
        "n_boot": args.n_boot,
        "slice_counts": {"cold": int((slice_of == "cold").sum()), "warm": int((slice_of == "warm").sum())},
        "results": {},
    }

    for method in METHODS:
        report["results"][method] = {}
        item_owner_arr = np.array(item_owner[method])
        item_row_arr = np.array(item_row[method])
        for slice_name, slice_mask in [("overall", np.ones(n_val, dtype=bool)),
                                        ("cold", slice_of == "cold"),
                                        ("warm", slice_of == "warm")]:
            slice_report = {}
            for metric in ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]:
                vals = np.array(metrics[method][metric])[slice_mask]
                slice_report[metric] = bootstrap_ci_mean(vals, n_boot=args.n_boot)

            global_idx_in_slice = np.where(slice_mask)[0]
            local_id = -np.ones(n_val, dtype=int)
            local_id[global_idx_in_slice] = np.arange(len(global_idx_in_slice))
            keep = local_id[item_owner_arr] >= 0
            owner_local = local_id[item_owner_arr[keep]]
            row_local = item_row_arr[keep]
            slice_report["coverage"] = bootstrap_ci_coverage(
                owner_local, row_local, len(global_idx_in_slice), n_docs, n_boot=args.n_boot)

            report["results"][method][slice_name] = slice_report

    # -- print summary ----------------------------------------------------------
    print("\n=== Q4 offline evaluation: BM25 vs LSA ===")
    for slice_name in ["overall", "cold", "warm"]:
        print(f"\n-- {slice_name} --")
        header = f"{'metric':<12}" + "".join(f"{m:>22}" for m in METHODS)
        print(header)
        for metric in ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]:
            row = f"{metric:<12}"
            for method in METHODS:
                r = report["results"][method][slice_name][metric]
                row += f"{r['mean']:>8.4f} [{r['ci_low']:.4f},{r['ci_high']:.4f}]"
            print(row)
        row = f"{'coverage':<12}"
        for method in METHODS:
            r = report["results"][method][slice_name]["coverage"]
            row += f"{r['point_estimate']:>8.4f} rs~N[{r['ci_low']:.4f},{r['ci_high']:.4f}]"
        print(row + "   (rs~N = with-replacement resampling band, see note in JSON; point_estimate is the reported figure)")

    out_path = os.path.join(args.data_dir, "reports", "mind_ranking_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
