#!/usr/bin/env python3
"""A2 Q3 -- ablation study, isolating which Q1 behavioural feature groups
actually drive the re-ranker's improvement over Assignment 1's fusion
score (the "baseline" for this assignment's Q3 -- see the project's
design note for why NRMS itself wasn't reproduced: a faithful
reproduction is a multi-day neural-training undertaking, not something
buildable-and-completable in this format, and "the MIND baseline" is the
other option Q3 explicitly allows).

Trains several LGBMRanker variants, each on a different SUBSET of
FEATURE_COLUMNS (retrieval.reranker_data), then reports AUC/MRR/nDCG@5/
nDCG@10 for each vs. the full-feature model, WITH a paired bootstrap 95%
CI on each variant's DROP relative to the full model -- a group whose
removal produces a CI that excludes zero contributed a real, not-noise
amount, exactly Q3's "claimed gains must ship a paired bootstrap 95% CI
that excludes zero" applied to ablation instead of the top-level
before/after comparison scripts/train_reranker.py already reports.

Variants (see ABLATION_GROUPS):
    full            -- every feature (the model scripts/train_reranker.py trains)
    retrieval_only  -- ONLY Assignment 1's bm25/semantic/fusion scores (no Q1 features at all)
    no_position     -- full minus candidate_position
    no_history_content -- full minus category/(sub)category-match/embedding-sim
    no_popularity   -- full minus article_train_ctr/article_train_impressions
    no_freshness    -- full minus the freshness proxy
    no_session      -- (EB-NeRD only) full minus session/engagement features

    python scripts/ablation_reranker.py --dataset mind --train_sample_size 200000
    python scripts/ablation_reranker.py --dataset ebnerd

Performance note: trains N variants, each paying the full LGBMRanker.fit()
cost -- use --train_sample_size (default 200,000) to keep this tractable;
without it, 6+ full-MIND-train fits would take hours, not minutes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bootstrap import bootstrap_ci_paired_delta
from retrieval.build_indices import load_indices
from retrieval.reranker_data import FEATURE_COLUMNS, load_split_features
from retrieval.reranker_eval import per_impression_metrics
from retrieval.text_utils import article_text

ABLATION_GROUPS = {
    "mind": {
        "retrieval_only": ["bm25_score", "semantic_score", "fusion_score"],
        "no_position": ["candidate_position"],
        "no_history_content": ["history_category_match", "history_subcategory_match", "history_embedding_sim"],
        "no_popularity": ["article_train_ctr", "article_train_impressions"],
        "no_freshness": ["article_days_since_first_seen"],
    },
    "ebnerd": {
        "retrieval_only": ["bm25_score", "semantic_score", "fusion_score"],
        "no_position": ["candidate_position"],
        "no_history_content": ["history_category_match", "history_embedding_sim"],
        "no_popularity": ["article_train_ctr", "article_train_impressions"],
        "no_freshness": ["article_days_since_publish"],
        "no_session": ["session_prior_click_count", "user_avg_past_read_time", "user_avg_past_scroll_pct"],
    },
}


def feature_set_for(dataset, variant):
    full = FEATURE_COLUMNS[dataset]
    if variant == "full":
        return full
    if variant == "retrieval_only":
        return ABLATION_GROUPS[dataset]["retrieval_only"]
    remove = set(ABLATION_GROUPS[dataset][variant])
    return [c for c in full if c not in remove]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--train_sample_size", type=int, default=200_000,
                     help="trains each of N variants on this many TRAIN impressions -- "
                          "keeps N full LGBMRanker.fit() calls tractable; val is always full")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import lightgbm as lgb

    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)
    bm25, semantic, config = load_indices(model_dir)
    recent_n, recency_decay = config["recent_n"], config["recency_decay"]
    fusion_alpha = 0.7
    all_feature_cols = FEATURE_COLUMNS[args.dataset]

    articles = pd.read_parquet(os.path.join(args.data_dir, "processed", args.dataset, "articles.parquet"))
    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))
    entity_lookup = dict(zip(articles["article_id"], articles["entities"]))

    variants = ["full", "retrieval_only", "no_position", "no_history_content", "no_popularity", "no_freshness"]
    if args.dataset == "ebnerd":
        variants.append("no_session")

    print(f"Dataset={args.dataset}, {len(variants)} variants: {variants}")
    print("Loading train split (reuses cached Assignment-1 retrieval scores if present) ...")
    train_df = load_split_features(args.dataset, "train", args.data_dir, bm25, semantic, text_lookup,
                                     entity_lookup, recent_n, recency_decay, fusion_alpha,
                                     sample_size=args.train_sample_size, seed=args.seed, feature_cols=all_feature_cols)
    print("Loading val split (full, not sampled) ...")
    val_df = load_split_features(args.dataset, "val", args.data_dir, bm25, semantic, text_lookup,
                                   entity_lookup, recent_n, recency_decay, fusion_alpha,
                                   feature_cols=all_feature_cols)
    print(f"  train: {len(train_df):,} rows / {train_df['impression_id'].nunique():,} impressions; "
          f"val: {len(val_df):,} rows / {val_df['impression_id'].nunique():,} impressions")

    y_train = train_df["label"].astype(int)
    group_train = train_df.groupby("impression_id", sort=False, observed=True).size().to_numpy()

    per_impression_by_variant = {}
    results = {}
    for variant in variants:
        cols = feature_set_for(args.dataset, variant)
        print(f"\n--- {variant} ({len(cols)} features: {cols}) ---")
        t0 = time.time()
        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=args.n_estimators, learning_rate=args.learning_rate,
            num_leaves=args.num_leaves, random_state=42, verbosity=-1,
        )
        model.fit(train_df[cols], y_train, group=group_train)
        print(f"  trained in {time.time()-t0:.1f}s")
        val_df[f"score_{variant}"] = model.predict(val_df[cols])
        per_imp = per_impression_metrics(val_df, f"score_{variant}")
        per_impression_by_variant[variant] = per_imp
        results[variant] = {
            "n_features": len(cols), "features": cols,
            "auc": float(np.nanmean(per_imp["auc"])), "mrr": float(np.nanmean(per_imp["mrr"])),
            "ndcg5": float(np.nanmean(per_imp["ndcg5"])), "ndcg10": float(np.nanmean(per_imp["ndcg10"])),
        }
        print(f"  auc={results[variant]['auc']:.4f} mrr={results[variant]['mrr']:.4f} "
              f"ndcg5={results[variant]['ndcg5']:.4f} ndcg10={results[variant]['ndcg10']:.4f}")

    # -- paired bootstrap: does removing this group cost something REAL? --
    full_metrics = per_impression_by_variant["full"]
    print(f"\n=== Ablation: {args.dataset} -- drop relative to 'full', paired bootstrap 95% CI ===")
    print(f"{'variant':<20}{'metric':<8}{'full':>10}{'variant':>10}{'drop':>10}{'CI':>24}{'real?':>7}")
    ablation_report = {"full": results["full"], "drops": {}}
    for variant in variants:
        if variant == "full":
            continue
        ablation_report["drops"][variant] = {}
        for metric in ["auc", "mrr", "ndcg5", "ndcg10"]:
            # "drop" = full - variant, so a POSITIVE drop means removing
            # the group hurt (full was better) -- bootstrap_ci_paired_delta
            # computes (after - before); passing (variant, full) as
            # (before, after) makes its "delta" exactly this drop.
            d = bootstrap_ci_paired_delta(per_impression_by_variant[variant][metric], full_metrics[metric],
                                            n_boot=args.n_boot, random_state=args.seed)
            ablation_report["drops"][variant][metric] = d
            ci_str = f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"
            real = "YES" if d["excludes_zero"] and d["delta"] > 0 else ("neg" if d["excludes_zero"] else "no")
            print(f"{variant:<20}{metric:<8}{results['full'][metric]:>10.4f}{results[variant][metric]:>10.4f}"
                  f"{d['delta']:>+10.4f}{ci_str:>24}{real:>7}")

    report = {"dataset": args.dataset, "train_sample_size": args.train_sample_size,
              "n_train_impressions": int(train_df["impression_id"].nunique()),
              "n_val_impressions": int(val_df["impression_id"].nunique()),
              "variants": results, "ablation_vs_full": ablation_report["drops"]}
    report_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_ablation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")
    print("\n'real?' column: YES = removing this group significantly HURT accuracy (CI excludes zero, "
          "positive drop) -- that group's contribution is real, not noise. 'no' = the CI includes zero, "
          "i.e. this group's contribution wasn't distinguishable from noise at this sample size.")


if __name__ == "__main__":
    main()
