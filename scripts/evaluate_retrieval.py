#!/usr/bin/env python3
"""Q2 + Q3: load the persisted BM25F + semantic (LSA or SBERT+FAISS) index
(built once by scripts/build_indices.py over the train+val article corpus
-- run that first), retrieve top-K candidates for each MINDsmall_dev
("val" split) impression using the user's recent click history as a
recency-weighted query, and report recall@K for K in {50, 100, 200} --
i.e. how often the impression's actual clicked article lands in the top-K
retrieved set. Also prints a lexical-vs-semantic comparison, overall and
sliced by cold-start vs. warm (Q3.5).

    python scripts/build_indices.py       # once, or whenever hyperparameters change
    python scripts/evaluate_retrieval.py

Reads data/processed/mind/interactions.parquet (built by build_pipeline.py)
and data/models/mind/{bm25.pkl,semantic.pkl,config.json} (built by
scripts/build_indices.py -- loading here, not re-fitting, is what
guarantees this script and evaluate_ranking.py score against the exact
same index instance). Writes data/reports/mind_retrieval_eval.json.

If the loaded semantic index is an SBERTIndex (semantic_backend="sbert"),
top-K retrieval goes through FAISS's native `search_topk` (no B x n_docs
dense matrix ever materialized); if it's an LSAIndex, it falls back to the
brute-force `score_batch_full` dense-matrix path, same as before.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.build_indices import load_indices
from retrieval.eval_utils import recall_hits_from_topk, recall_hits_matrix
from retrieval.lsa import mean_pool_user_vector
from retrieval.text_utils import article_text, recency_weights, weighted_query_terms

KS = [50, 100, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--recent_n", type=int, default=None,
                     help="defaults to the value baked into the loaded index's config.json")
    ap.add_argument("--recency_decay", type=float, default=None,
                     help="defaults to the value baked into the loaded index's config.json")
    ap.add_argument("--cold_start_threshold", type=int, default=5,
                     help="impressions with fewer than this many history articles are the 'cold' slice "
                          "(same convention as scripts/evaluate_ranking.py, for cross-question consistency)")
    args = ap.parse_args()
    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)

    bm25, semantic, config = load_indices(model_dir)
    semantic_backend = config.get("semantic_backend", "lsa")
    use_faiss = hasattr(semantic, "search_topk")
    recent_n = args.recent_n if args.recent_n is not None else config["recent_n"]
    recency_decay = args.recency_decay if args.recency_decay is not None else config["recency_decay"]
    print(f"Loaded index from {model_dir}: {config['n_docs']:,} docs, "
          f"k1={config['bm25_k1']}, b={config['bm25_b']}, field_weights={config['bm25_field_weights']}, "
          f"semantic_backend={semantic_backend} (faiss={use_faiss}), entity_fusion={config['entity_fusion']}, "
          f"recent_n={recent_n}, recency_decay={recency_decay}")

    proc_dir = os.path.join(args.data_dir, "processed", args.dataset)
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    val = interactions[interactions["split"] == "val"].reset_index(drop=True)
    print(f"Loaded {len(articles):,} articles, {len(val):,} val impressions")

    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))

    truths = [
        {cid for cid, lab in zip(cands, labs) if lab == 1}
        for cands, labs in zip(val["candidate_article_ids"], val["labels"])
    ]
    n_evaluable = sum(1 for t in truths if t)
    print(f"{n_evaluable:,} / {len(val):,} val impressions have a ground-truth click")

    # -- Q3.5 slicing: cold-start (few history articles) vs. warm -------------
    hist_lens = val["history_article_ids"].apply(len).to_numpy()
    slice_of = np.where(hist_lens < args.cold_start_threshold, "cold", "warm")
    print(f"Slicing @ history_length < {args.cold_start_threshold}: "
          f"{(slice_of == 'cold').sum():,} cold, {(slice_of == 'warm').sum():,} warm")

    # -- Q2.3/Q2.4 + Q3.3/Q3.4: batched top-K retrieval + recall@K -------------
    n = len(val)
    max_k = max(KS)
    bm25_hits = np.full((n, len(KS)), np.nan)
    semantic_hits = np.full((n, len(KS)), np.nan)
    t_start = time.time()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch_hist = val["history_article_ids"].iloc[start:end].tolist()
        batch_truths = truths[start:end]

        bm25_query_weights, recent_lists = [], []
        for h in batch_hist:
            qw = weighted_query_terms(h, text_lookup, recent_n=recent_n, decay=recency_decay)
            bm25_query_weights.append(qw)
            recent_lists.append(list(h)[-recent_n:] if len(h) else [])

        bm25_scores = bm25.score_batch_full(bm25_query_weights)
        bm25_hits[start:end, :] = recall_hits_matrix(bm25_scores, bm25.id_to_row, batch_truths, KS)

        user_vecs = []
        for recent in recent_lists:
            embs = [semantic.get_embedding(a) for a in recent]
            w = recency_weights(len(recent), recency_decay) if recent else None
            user_vecs.append(mean_pool_user_vector(embs, weights=w))
        emb_dim = semantic.embeddings.shape[1]
        user_mat = np.array([v if v is not None else np.zeros(emb_dim) for v in user_vecs])

        if use_faiss:
            _, topk_idx = semantic.search_topk(user_mat, k=min(max_k, config["n_docs"]))
            semantic_hits[start:end, :] = recall_hits_from_topk(topk_idx, semantic.doc_ids, batch_truths, KS)
        else:
            semantic_scores = semantic.score_batch_full(user_mat)
            semantic_hits[start:end, :] = recall_hits_matrix(semantic_scores, semantic.id_to_row, batch_truths, KS)

        if (start // args.batch_size) % 20 == 0:
            print(f"  {end:,}/{n:,} impressions scored ({time.time()-t_start:.1f}s elapsed)")

    def slice_recall(hits, mask):
        with np.errstate(invalid="ignore"):
            vals = np.nanmean(hits[mask, :], axis=0)
        n_eval = np.sum(~np.isnan(hits[mask, :]), axis=0)
        return {str(k): (float(vals[i]) if n_eval[i] > 0 else float("nan")) for i, k in enumerate(KS)}, \
               {str(k): int(n_eval[i]) for i, k in enumerate(KS)}

    slices = {"overall": np.ones(n, dtype=bool), "cold": slice_of == "cold", "warm": slice_of == "warm"}
    recall_by_slice = {"bm25": {}, "semantic": {}}
    n_eval_by_slice = {"bm25": {}, "semantic": {}}
    for slice_name, mask in slices.items():
        recall_by_slice["bm25"][slice_name], n_eval_by_slice["bm25"][slice_name] = slice_recall(bm25_hits, mask)
        recall_by_slice["semantic"][slice_name], n_eval_by_slice["semantic"][slice_name] = slice_recall(semantic_hits, mask)

    print(f"\n=== Recall@K: BM25F (lexical) vs {semantic_backend} (semantic) ===")
    for slice_name in ["overall", "cold", "warm"]:
        print(f"\n-- {slice_name} ({int(slices[slice_name].sum()):,} impressions) --")
        print(f"{'K':>5} {'BM25F':>10} {semantic_backend:>10}")
        for k in KS:
            b = recall_by_slice["bm25"][slice_name][str(k)]
            s = recall_by_slice["semantic"][slice_name][str(k)]
            print(f"{k:>5} {b:>10.4f} {s:>10.4f}")

    better_by_slice = {
        slice_name: {str(k): ("bm25" if recall_by_slice["bm25"][slice_name][str(k)] >=
                               recall_by_slice["semantic"][slice_name][str(k)] else semantic_backend)
                     for k in KS}
        for slice_name in slices
    }
    print(f"\nBetter method per (slice, K): {better_by_slice}")

    report = {
        "n_val_impressions": int(len(val)),
        "n_evaluable_impressions": int(n_evaluable),
        "index_config": config,
        "semantic_backend": semantic_backend,
        "recent_n_used": recent_n,
        "recency_decay_used": recency_decay,
        "cold_start_threshold": args.cold_start_threshold,
        "slice_counts": {s: int(m.sum()) for s, m in slices.items()},
        "recall_at_k": recall_by_slice,
        "n_evaluable_at_k": n_eval_by_slice,
        "better_method_per_slice_k": better_by_slice,
    }
    out_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_retrieval_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
