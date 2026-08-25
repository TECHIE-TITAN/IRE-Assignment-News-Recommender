#!/usr/bin/env python3
"""k1/b hyperparameter sweep for BM25F (Q2), evaluated with the same
restricted-candidate scoring Q4/Q5 use (`score_candidates` -- rank a given
impression's own candidates), not Q2/Q3's full-corpus retrieval. A grid
point only needs to know how well BM25F *ranks* an impression's own
candidates -- ~150x cheaper per impression than full-corpus recall@K, and
it's what actually matters for Q4/Q5 scoring (and the Codabench score),
so it's the right metric to tune against.

Runs on a fixed-seed random subsample of val impressions (not the full
~376K) so the whole grid finishes in a few minutes instead of ~150x
(grid size) minutes -- a standard, defensible practice for hyperparameter
search where you don't need final-report precision, just enough signal to
pick a good (k1, b).

    python scripts/tune_bm25.py

Reads data/processed/mind/{articles,interactions}.parquet. Writes
data/reports/mind_bm25_tuning.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25 import BM25Index
from retrieval.ranking_metrics import auc_score, mrr_score, ndcg_at_k
from retrieval.text_utils import article_text, weighted_query_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--sample_size", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recent_n", type=int, default=20)
    ap.add_argument("--recency_decay", type=float, default=0.85)
    ap.add_argument("--k1_grid", type=float, nargs="+", default=[1.0, 1.2, 1.5, 1.8, 2.2])
    ap.add_argument("--b_grid", type=float, nargs="+", default=[0.3, 0.5, 0.75, 0.9])
    args = ap.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed", args.dataset)
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    val = interactions[interactions["split"] == "val"].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(val), size=min(args.sample_size, len(val)), replace=False)
    sample = val.iloc[sample_idx].reset_index(drop=True)
    print(f"Tuning on a fixed-seed subsample of {len(sample):,} / {len(val):,} val impressions "
          f"(seed={args.seed}); field_weights held at BM25Index defaults (title=2.0, abstract=1.0)")

    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))
    titles = articles["title"].tolist()
    abstracts = articles["abstract"].tolist()
    doc_ids = articles["article_id"].tolist()

    hist_list = sample["history_article_ids"].tolist()
    cand_list = sample["candidate_article_ids"].tolist()
    label_list = sample["labels"].tolist()

    # Query weights don't depend on k1/b -- build once, reuse across the grid.
    query_weights = [weighted_query_terms(h, text_lookup, recent_n=args.recent_n, decay=args.recency_decay)
                      for h in hist_list]

    results = []
    t_grid = time.time()
    for k1 in args.k1_grid:
        for b in args.b_grid:
            t0 = time.time()
            bm25 = BM25Index(k1=k1, b=b).fit(doc_ids, titles, abstracts)
            aucs, mrrs, ndcg10s = [], [], []
            for qw, cand_ids, labels in zip(query_weights, cand_list, label_list):
                labels_arr = np.array(labels)
                scores = bm25.score_candidates(qw, cand_ids)
                aucs.append(auc_score(labels_arr, scores))
                mrrs.append(mrr_score(labels_arr, scores))
                ndcg10s.append(ndcg_at_k(labels_arr, scores, 10))
            row = {
                "k1": k1, "b": b,
                "auc": float(np.nanmean(aucs)),
                "mrr": float(np.nanmean(mrrs)),
                "ndcg10": float(np.nanmean(ndcg10s)),
                "seconds": round(time.time() - t0, 1),
            }
            results.append(row)
            print(f"  k1={k1:>4} b={b:>4}  AUC={row['auc']:.4f}  MRR={row['mrr']:.4f}  "
                  f"nDCG@10={row['ndcg10']:.4f}  ({row['seconds']:.1f}s)")

    best = max(results, key=lambda r: r["auc"])
    default = next((r for r in results if r["k1"] == 1.5 and r["b"] == 0.75), None)
    print(f"\nGrid finished in {time.time()-t_grid:.1f}s. Best by AUC: k1={best['k1']}, b={best['b']} "
          f"(AUC={best['auc']:.4f})")
    if default:
        print(f"Textbook default (k1=1.5, b=0.75): AUC={default['auc']:.4f} "
              f"(delta vs best: {best['auc']-default['auc']:+.4f})")

    out_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_bm25_tuning.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"sample_size": len(sample), "seed": args.seed, "recency_decay": args.recency_decay,
                    "results": results, "best_by_auc": best, "default_k1_b": default}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
