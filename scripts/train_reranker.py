#!/usr/bin/env python3
"""A2 Q2 -- two-stage retrieve-then-rank: trains a GBDT re-ranker (LightGBM
`LGBMRanker`, lambdarank objective -- Q2's "Option A") over Assignment 1's
own retrieval scores (retrieval/reranker_scores.py) plus Assignment 2 Q1's
engineered behavioural features (pipeline/features_mind.py /
pipeline/features_ebnerd.py, via scripts/build_features.py), then reports
AUC/MRR/nDCG@5/nDCG@10 on val BEFORE (Assignment 1's fusion score alone)
and AFTER (the trained re-ranker) re-ranking.

`lambdarank`, not plain binary classification: Q2 is fundamentally a
*ranking* problem (order candidates well within each impression), not an
independent per-candidate classification problem -- lambdarank optimizes
directly for a ranking objective (pairwise, within each impression's own
`group`), which is what AUC/MRR/nDCG actually measure. A plain classifier
(e.g. logistic regression / binary GBDT) would optimize per-row log-loss
without ever knowing which rows belong to the same impression.

Deliberately dataset-specific FEATURE_COLUMNS (MIND vs EB-NeRD have
genuinely different Q1 feature sets, see pipeline/features_{mind,ebnerd}.py)
-- only the Assignment-1 retrieval-score columns (bm25_score/semantic_score/
fusion_score) and the training/eval mechanics are shared.

Reads:
    data/feature_store/<dataset>/candidate_features_{train,val}.parquet  (A2 Q1)
    data/processed/<dataset>/{articles,interactions}.parquet             (A1 Q1)
    data/models/<dataset>/{bm25.pkl,semantic.pkl,config.json}            (A1 Q2/Q3)
Writes:
    data/models/<dataset>/reranker.pkl + reranker_config.json
    data/reports/<dataset>_reranker_eval.json

    python scripts/train_reranker.py --dataset mind
    python scripts/train_reranker.py --dataset ebnerd

Performance note: computing Assignment-1 retrieval scores per impression
(needed as re-ranker features here) is a second, separate per-impression
pass over each split -- the same big-O shape and cost as
scripts/evaluate_ranking.py's own scoring loop, paid again here rather than
reused, since Q1's feature table doesn't itself store these scores. An
earlier version of this script accumulated that entire pass as one Python
list before ever building a DataFrame, which OOM-killed on full MIND train
(2.2M impressions) after ~2.2 hours, running at ~277 impr/sec -- 6-15x
slower than Assignment 1's ~1,650-4,300 impr/sec for the identical
per-impression work, because the growing list of tens of millions of boxed
Python tuples drove the process into swap well before it actually ran out
of memory outright. Fixed by writing results incrementally to a cached
parquet file in bounded chunks (--chunk_size) instead -- see
`compute_retrieval_score_table`. Still worth a --limit sanity run before
committing to a full pass on a new machine/dataset combination, since the
fix has only been reasoned through, not yet re-benchmarked end-to-end.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.build_indices import load_indices
from retrieval.ranking_metrics import auc_score, mrr_score, ndcg_at_k
from retrieval.reranker_scores import compute_retrieval_scores
from retrieval.text_utils import article_text

FEATURE_COLUMNS = {
    "mind": [
        "candidate_position", "history_length", "history_category_match",
        "history_subcategory_match", "history_embedding_sim",
        "article_train_ctr", "article_train_impressions", "article_days_since_first_seen",
        "bm25_score", "semantic_score", "fusion_score",
    ],
    "ebnerd": [
        "candidate_position", "history_length", "history_category_match",
        "history_embedding_sim", "article_train_ctr", "article_train_impressions",
        "article_days_since_publish", "session_prior_click_count",
        "user_avg_past_read_time", "user_avg_past_scroll_pct",
        "bm25_score", "semantic_score", "fusion_score",
    ],
}


def compute_retrieval_score_table(split_interactions, bm25, semantic, text_lookup, entity_lookup,
                                    recent_n, recency_decay, fusion_alpha, cache_path, chunk_size=20_000):
    """One row per (impression_id, candidate_article_id) with Assignment 1's
    bm25/semantic/fusion scores -- merged onto Q1's behavioural feature
    table by the caller. Written incrementally to `cache_path` in bounded
    chunks (mirroring scripts/build_features.py's own chunked-write
    pattern), NOT accumulated as one giant Python list across the whole
    split -- an earlier version did that and it's what caused a real,
    observed failure on full MIND train (2.2M impressions): the unbounded
    list of tens of millions of boxed-Python tuples drove the process into
    swap, which explains both the ~6-15x slower-than-expected throughput
    (~277 impr/sec measured vs. Assignment 1's ~1,650-4,300 impr/sec for
    the identical per-impression scoring work) and the eventual OOM kill.
    Also doubles as a cache: this step alone took over 2 hours on full MIND
    train, so a re-run (e.g. to retrain with different LightGBM
    hyperparameters) reuses the persisted file instead of recomputing it --
    `cache_path` itself encodes recent_n/recency_decay/fusion_alpha (see
    caller) so a changed hyperparameter can't silently reuse a stale one."""
    if os.path.isfile(cache_path):
        print(f"  reusing cached retrieval scores -> {cache_path}")
        return pd.read_parquet(cache_path)

    n = len(split_interactions)
    t0 = time.time()
    writer = None
    n_written = 0
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    for start in range(0, n, chunk_size):
        chunk_rows = []
        for imp in split_interactions.iloc[start:start + chunk_size].itertuples(index=False):
            cand_ids = imp.candidate_article_ids
            if len(cand_ids) == 0:
                continue
            bm25_scores, semantic_scores, fusion_scores = compute_retrieval_scores(
                imp.history_article_ids, cand_ids, bm25, semantic, text_lookup, entity_lookup,
                recent_n, recency_decay, fusion_alpha,
            )
            for cid, b, s, f in zip(cand_ids, bm25_scores, semantic_scores, fusion_scores):
                chunk_rows.append((imp.impression_id, cid, float(b), float(s), float(f)))
        if not chunk_rows:
            continue
        chunk_df = pd.DataFrame(chunk_rows, columns=[
            "impression_id", "candidate_article_id", "bm25_score", "semantic_score", "fusion_score",
        ])
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(cache_path, table.schema)
        writer.write_table(table)
        n_written += len(chunk_df)
        done = min(start + chunk_size, n)
        print(f"    retrieval scores: {done:,}/{n:,} impressions -> {n_written:,} rows written "
              f"({time.time()-t0:.1f}s elapsed)")
    if writer is not None:
        writer.close()
    return pd.read_parquet(cache_path)


def load_split_features(dataset, split, data_dir, bm25, semantic, text_lookup, entity_lookup,
                          recent_n, recency_decay, fusion_alpha, limit=None, chunk_size=20_000,
                          sample_size=None, seed=42, feature_cols=None):
    feat_path = os.path.join(data_dir, "feature_store", dataset, f"candidate_features_{split}.parquet")
    if not os.path.isfile(feat_path):
        raise SystemExit(f"{feat_path} not found -- run `python scripts/build_features.py "
                          f"--dataset {dataset} --split {split}` first (A2 Q1).")
    feats = pd.read_parquet(feat_path)

    interactions = pd.read_parquet(os.path.join(data_dir, "processed", dataset, "interactions.parquet"))
    split_interactions = interactions[interactions["split"] == split].reset_index(drop=True)
    if limit:
        keep_ids = set(feats["impression_id"].unique()[:limit])
        feats = feats[feats["impression_id"].isin(keep_ids)]
        split_interactions = split_interactions[split_interactions["impression_id"].isin(keep_ids)]

    print(f"  computing Assignment-1 retrieval scores for {len(split_interactions):,} {split} impressions ...")
    cache_path = os.path.join(
        data_dir, "feature_store", dataset,
        f"retrieval_scores_{split}_n{recent_n}_d{recency_decay}_a{fusion_alpha}"
        + (f"_limit{limit}" if limit else "") + ".parquet",
    )
    score_table = compute_retrieval_score_table(
        split_interactions, bm25, semantic, text_lookup, entity_lookup, recent_n, recency_decay, fusion_alpha,
        cache_path, chunk_size=chunk_size,
    )

    if sample_size:
        # Applied AFTER the (possibly-cached) retrieval-score computation,
        # not before: this reuses an already-computed full-split cache
        # rather than forcing a smaller one to be recomputed from scratch
        # every time --train_sample_size changes. Applied BEFORE the merge
        # below, though, since that join (not the cache lookup) is what
        # actually OOM-killed on full MIND train (83.5M rows both sides) --
        # see the module docstring. A fixed seed keeps this reproducible.
        rng = np.random.default_rng(seed)
        all_ids = feats["impression_id"].unique()
        if sample_size < len(all_ids):
            keep_ids = set(rng.choice(all_ids, size=sample_size, replace=False))
            feats = feats[feats["impression_id"].isin(keep_ids)]
            score_table = score_table[score_table["impression_id"].isin(keep_ids)]
            print(f"  sampled down to {sample_size:,}/{len(all_ids):,} {split} impressions "
                  f"(--train_sample_size, seed={seed})")

    # Merging on `category` dtype instead of raw strings is both faster and
    # much lower peak memory for tens-of-millions-of-rows joins (pandas
    # hashes/compares the small set of category codes, not full strings
    # repeated millions of times) -- a real, always-beneficial change, not
    # scale-dependent like sampling.
    for col in ["impression_id", "candidate_article_id"]:
        feats[col] = feats[col].astype("category")
        score_table[col] = score_table[col].astype("category")
    n_feats = len(feats)
    merged = feats.merge(score_table, on=["impression_id", "candidate_article_id"], how="inner")
    del feats, score_table
    # inner join, not left: a candidate present in the A2 Q1 feature table
    # but missing a retrieval score (e.g. an empty candidate list edge case
    # skipped above) can't be trained/scored on anyway -- dropping it here
    # is louder than silently carrying NaN feature columns into LightGBM.
    if len(merged) != n_feats:
        print(f"  NOTE: {n_feats - len(merged):,}/{n_feats:,} feature rows had no matching "
              f"retrieval score and were dropped (see inner-join note in load_split_features).")

    if feature_cols:
        # float32 halves the numeric footprint of the columns LightGBM
        # actually trains on vs. pandas' float64 default -- GBDTs don't
        # need float64 precision, and at tens-of-millions-of-rows scale
        # this is a real, non-cosmetic memory saving, not premature
        # optimization.
        for col in feature_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype("float32")

    return merged.sort_values("impression_id", kind="stable").reset_index(drop=True)


def evaluate_scores(df, score_col, label_col="label"):
    """Per-impression AUC/MRR/nDCG@5/nDCG@10, nan-averaged across
    impressions -- auc_score/mrr_score/ndcg_at_k already return NaN for an
    impression with no positive label (AUC/MRR/nDCG are undefined there),
    so `np.nanmean` is the correct aggregation, not a workaround."""
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for _, g in df.groupby("impression_id", sort=False, observed=True):
        labels = g[label_col].to_numpy()
        scores = g[score_col].to_numpy()
        aucs.append(auc_score(labels, scores))
        mrrs.append(mrr_score(labels, scores))
        ndcg5s.append(ndcg_at_k(labels, scores, 5))
        ndcg10s.append(ndcg_at_k(labels, scores, 10))
    return {
        "auc": float(np.nanmean(aucs)), "mrr": float(np.nanmean(mrrs)),
        "ndcg5": float(np.nanmean(ndcg5s)), "ndcg10": float(np.nanmean(ndcg10s)),
        "n_impressions_scored": int(len(aucs)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--recent_n", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--recency_decay", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--fusion_alpha", type=float, default=0.7, help="see retrieval/fusion.py")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--limit", type=int, default=None, help="cap on impressions per split, for a quick test run")
    ap.add_argument("--chunk_size", type=int, default=20_000,
                     help="impressions per chunk when computing Assignment-1 retrieval scores -- "
                          "bounds memory during that step, see compute_retrieval_score_table")
    ap.add_argument("--train_sample_size", type=int, default=None,
                     help="train on a random sample of this many TRAIN impressions instead of all of "
                          "them (val is always evaluated in full, regardless) -- recommended at MIND "
                          "scale (2.2M train impressions / 83.5M candidate rows OOM-killed the merge/fit "
                          "step even after the retrieval-score computation itself was fixed to be "
                          "memory-bounded; see the module docstring). A no-op if the split already has "
                          "fewer impressions than this (e.g. EB-NeRD train, 232K impressions).")
    ap.add_argument("--seed", type=int, default=42, help="for --train_sample_size's sampling")
    args = ap.parse_args()

    import lightgbm as lgb

    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)
    bm25, semantic, config = load_indices(model_dir)
    recent_n = args.recent_n if args.recent_n is not None else config["recent_n"]
    recency_decay = args.recency_decay if args.recency_decay is not None else config["recency_decay"]
    feature_cols = FEATURE_COLUMNS[args.dataset]
    print(f"Dataset={args.dataset}, recent_n={recent_n}, recency_decay={recency_decay}, "
          f"fusion_alpha={args.fusion_alpha}, {len(feature_cols)} features: {feature_cols}")

    articles = pd.read_parquet(os.path.join(args.data_dir, "processed", args.dataset, "articles.parquet"))
    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))
    entity_lookup = dict(zip(articles["article_id"], articles["entities"]))

    print("Loading train split ...")
    train_df = load_split_features(args.dataset, "train", args.data_dir, bm25, semantic, text_lookup,
                                     entity_lookup, recent_n, recency_decay, args.fusion_alpha, args.limit,
                                     args.chunk_size, args.train_sample_size, args.seed, feature_cols)
    print("Loading val split ...")
    val_df = load_split_features(args.dataset, "val", args.data_dir, bm25, semantic, text_lookup,
                                   entity_lookup, recent_n, recency_decay, args.fusion_alpha, args.limit,
                                   args.chunk_size, None, args.seed, feature_cols)
    print(f"  train: {len(train_df):,} rows / {train_df['impression_id'].nunique():,} impressions; "
          f"val: {len(val_df):,} rows / {val_df['impression_id'].nunique():,} impressions")

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    group_train = train_df.groupby("impression_id", sort=False, observed=True).size().to_numpy()

    print(f"Training LGBMRanker (lambdarank, n_estimators={args.n_estimators}, "
          f"learning_rate={args.learning_rate}, num_leaves={args.num_leaves}) ...")
    t0 = time.time()
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        num_leaves=args.num_leaves, random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train, group=group_train)
    print(f"  trained in {time.time()-t0:.1f}s")

    # No .copy() here: val_df is already a fresh object from
    # load_split_features (not a view of anything else), and at MIND val
    # scale (~14M rows) an unnecessary copy is a real, avoidable transient
    # memory cost right when LightGBM's own training-time structures may
    # still be alive.
    val_df["reranker_score"] = model.predict(val_df[feature_cols])

    before = evaluate_scores(val_df, "fusion_score")
    after = evaluate_scores(val_df, "reranker_score")
    print("\n=== Q2: before (Assignment-1 fusion score) vs after (LGBMRanker re-ranker) ===")
    print(f"{'metric':<10}{'before':>12}{'after':>12}{'delta':>12}")
    for m in ["auc", "mrr", "ndcg5", "ndcg10"]:
        print(f"{m:<10}{before[m]:>12.4f}{after[m]:>12.4f}{after[m]-before[m]:>+12.4f}")

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "reranker.pkl"), "wb") as f:
        pickle.dump(model, f)
    reranker_config = {
        "dataset": args.dataset, "feature_columns": feature_cols, "recent_n": recent_n,
        "recency_decay": recency_decay, "fusion_alpha": args.fusion_alpha,
        "n_estimators": args.n_estimators, "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
    }
    with open(os.path.join(model_dir, "reranker_config.json"), "w") as f:
        json.dump(reranker_config, f, indent=2)

    report = {
        "dataset": args.dataset, "config": reranker_config,
        "n_train_impressions": int(train_df["impression_id"].nunique()),
        "n_val_impressions": int(val_df["impression_id"].nunique()),
        "before_fusion_score": before, "after_reranker": after,
    }
    report_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_reranker_eval.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {model_dir}/reranker.pkl, {model_dir}/reranker_config.json, {report_path}")


if __name__ == "__main__":
    main()
