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
retrieval/reranker_data.py's `compute_retrieval_score_table` (shared with
scripts/evaluate_reranker.py, which needs the identical data-loading path).
Still worth a --limit sanity run before committing to a full pass on a new
machine/dataset combination, since the fix has only been reasoned through,
not yet re-benchmarked end-to-end.
"""

import argparse
import json
import os
import pickle
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.build_indices import load_indices
from retrieval.reranker_data import FEATURE_COLUMNS, load_split_features
from retrieval.reranker_eval import evaluate_scores
from retrieval.text_utils import article_text


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
