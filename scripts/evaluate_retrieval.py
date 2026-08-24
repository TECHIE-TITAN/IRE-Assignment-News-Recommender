#!/usr/bin/env python3
"""Q2 + Q3: build a BM25 index and an LSA (TF-IDF+SVD) index over the
MIND train+val article corpus, retrieve top-K candidates for each
MINDsmall_dev ("val" split) impression using the user's recent click
history as the query, and report recall@K for K in {50, 100, 200} --
i.e. how often the impression's actual clicked article lands in the
top-K retrieved set. Also prints a lexical-vs-semantic comparison (Q3.5).

    python scripts/evaluate_retrieval.py

Reads data/processed/mind/{articles,interactions}.parquet (built by
build_pipeline.py). Writes data/reports/mind_retrieval_eval.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.build_indices import fit_indices
from retrieval.eval_utils import recall_at_k_from_scores
from retrieval.lsa import mean_pool_user_vector
from retrieval.text_utils import build_query_text, tokenize

KS = [50, 100, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--recent_n", type=int, default=20, help="how many recent history articles to build the query from")
    ap.add_argument("--lsa_components", type=int, default=128)
    args = ap.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed", "mind")
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    val = interactions[interactions["split"] == "val"].reset_index(drop=True)
    print(f"Loaded {len(articles):,} articles, {len(val):,} val impressions")

    title_lookup = dict(zip(articles["article_id"], articles["title"]))

    # -- Q2.1 / Q3.1: build both indices over the same corpus -----------------
    bm25, lsa, doc_ids = fit_indices(articles, lsa_components=args.lsa_components)

    # -- Q2.2 / Q3.3: build per-impression queries -----------------------------
    def query_tokens_for(history_ids):
        recent = list(history_ids)[-args.recent_n:] if len(history_ids) else []
        return tokenize(build_query_text(history_ids, title_lookup, recent_n=args.recent_n)), recent

    truths = [
        {cid for cid, lab in zip(cands, labs) if lab == 1}
        for cands, labs in zip(val["candidate_article_ids"], val["labels"])
    ]
    n_evaluable = sum(1 for t in truths if t)
    print(f"{n_evaluable:,} / {len(val):,} val impressions have a ground-truth click")

    # -- Q2.3/Q2.4 + Q3.3/Q3.4: batched top-K retrieval + recall@K -------------
    bm25_hits = {k: [] for k in KS}
    lsa_hits = {k: [] for k in KS}
    n = len(val)
    t_start = time.time()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch_hist = val["history_article_ids"].iloc[start:end].tolist()
        batch_truths = truths[start:end]

        bm25_query_tokens, recent_lists = [], []
        for h in batch_hist:
            qt, recent = query_tokens_for(h)
            bm25_query_tokens.append(qt)
            recent_lists.append(recent)

        bm25_scores = bm25.score_batch_full(bm25_query_tokens)
        r, _ = recall_at_k_from_scores(bm25_scores, bm25.id_to_row, batch_truths, KS)
        for k in KS:
            if not np.isnan(r[k]):
                bm25_hits[k].append((r[k], sum(1 for t in batch_truths if t)))

        user_vecs = []
        for recent in recent_lists:
            embs = [lsa.get_embedding(a) for a in recent]
            user_vecs.append(mean_pool_user_vector(embs))
        user_mat = np.array([v if v is not None else np.zeros(args.lsa_components) for v in user_vecs])
        lsa_scores = lsa.score_batch_full(user_mat)
        r, _ = recall_at_k_from_scores(lsa_scores, lsa.id_to_row, batch_truths, KS)
        for k in KS:
            if not np.isnan(r[k]):
                lsa_hits[k].append((r[k], sum(1 for t in batch_truths if t)))

        if (start // args.batch_size) % 20 == 0:
            print(f"  {end:,}/{n:,} impressions scored ({time.time()-t_start:.1f}s elapsed)")

    def weighted_avg(pairs):
        total_n = sum(cnt for _, cnt in pairs)
        return (sum(r * cnt for r, cnt in pairs) / total_n) if total_n else float("nan")

    bm25_recall = {k: weighted_avg(v) for k, v in bm25_hits.items()}
    lsa_recall = {k: weighted_avg(v) for k, v in lsa_hits.items()}

    print("\n=== Recall@K: BM25 (lexical) vs LSA (semantic) ===")
    print(f"{'K':>5} {'BM25':>10} {'LSA':>10}")
    for k in KS:
        print(f"{k:>5} {bm25_recall[k]:>10.4f} {lsa_recall[k]:>10.4f}")

    better = {k: ("bm25" if bm25_recall[k] >= lsa_recall[k] else "lsa") for k in KS}
    print(f"\nBetter method per K: {better}")

    report = {
        "n_val_impressions": int(len(val)),
        "n_evaluable_impressions": int(n_evaluable),
        "recent_n_history": args.recent_n,
        "lsa_components": args.lsa_components,
        "recall_at_k": {"bm25": bm25_recall, "lsa": lsa_recall},
        "better_method_per_k": better,
    }
    out_path = os.path.join(args.data_dir, "reports", "mind_retrieval_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
