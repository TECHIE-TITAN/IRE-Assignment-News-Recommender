#!/usr/bin/env python3
"""A2 Q1 -- builds the candidate-level behavioural feature table a re-ranker
(A2 Q2) will train/score on: one row per (impression_id, candidate_article_id),
with the click-history, article, and (EB-NeRD only) session/engagement
features documented in pipeline/features_mind.py / pipeline/features_ebnerd.py.

Deliberately dataset-specific under the hood (`--dataset` dispatches to two
separate feature modules with two separate column sets) -- MIND and EB-NeRD
have genuinely different available signals (session/dwell time, true
publish timestamps), so this does NOT force one shared feature schema the
way pipeline/schema.py's article/interaction tables are shared. Only the
CLI shape and the output-parquet mechanics are common.

Reads:
    data/processed/<dataset>/interactions.parquet   (from build_pipeline.py)
    data/processed/<dataset>/articles.parquet       (from build_pipeline.py)
    data/feature_store/<dataset>/article_features.parquet  (from build_pipeline.py, train-only CTR)
    data/models/<dataset>/semantic.pkl + config.json        (from scripts/build_indices.py)
Writes:
    data/feature_store/<dataset>/candidate_features_<split>.parquet

    python scripts/build_features.py --dataset mind --split val
    python scripts/build_features.py --dataset mind --split train
    python scripts/build_features.py --dataset ebnerd --split val
    python scripts/build_features.py --dataset ebnerd --split train

Performance note (read before running on the full MIND train split): the
per-impression loop in pipeline/features_{mind,ebnerd}.py is a straightforward
first implementation (Python-level, not vectorized like Assignment 1's
sparse-matrix BM25 scoring) -- fine for val (376K MIND impressions / 245K
EB-NeRD), but MIND's train split (2.2M impressions, tens of millions of
candidate rows once exploded) will take a while. Use `--limit` to test on a
small slice first and `--chunk_size` to bound memory; this has NOT been
benchmarked end-to-end on full MIND train, so treat the first real run as a
timing measurement, not an assumption.
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ebnerd import load_articles as load_ebnerd_articles_raw
from pipeline.features_common import compute_first_seen_times
from pipeline.features_ebnerd import build_ebnerd_candidate_features, precompute_session_engagement
from pipeline.features_mind import build_mind_candidate_features
from retrieval.build_indices import load_indices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--out_dir", default=None, help="defaults to data/feature_store/<dataset>")
    ap.add_argument("--recent_n", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--recency_decay", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--chunk_size", type=int, default=20_000, help="impressions processed per chunk")
    ap.add_argument("--limit", type=int, default=None, help="cap on number of impressions (for a quick test run)")
    ap.add_argument("--ebnerd_bundle", default="ebnerd_small", help="EB-NeRD only, raw dir name for session/publish data")
    args = ap.parse_args()

    if args.dataset == "mind" and args.split == "test":
        raise SystemExit("MINDlarge_test has no click labels -- feature building against it isn't "
                          "meaningful for training/eval; use it only at A2 Q2's prediction-generation step.")
    if args.dataset == "ebnerd" and args.split == "test":
        raise SystemExit("ebnerd_small has no test split in this project's scope (see build_pipeline.py); "
                          "use --split train or val.")

    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)
    out_dir = args.out_dir or os.path.join(args.data_dir, "feature_store", args.dataset)
    proc_dir = os.path.join(args.data_dir, "processed", args.dataset)

    print(f"Loading {args.dataset} processed tables and Assignment-1 semantic index ...")
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    article_pop = pd.read_parquet(
        os.path.join(args.data_dir, "feature_store", args.dataset, "article_features.parquet")
    ).set_index("article_id")[["train_ctr", "train_impressions"]]
    _, semantic, config = load_indices(model_dir)

    recent_n = args.recent_n if args.recent_n is not None else config.get("recent_n", 20)
    recency_decay = args.recency_decay if args.recency_decay is not None else config.get("recency_decay", 0.85)
    print(f"  recent_n={recent_n}, recency_decay={recency_decay} (from {'CLI' if args.recent_n else 'config.json'})")

    split_interactions = interactions[interactions["split"] == args.split].reset_index(drop=True)
    if args.limit:
        split_interactions = split_interactions.head(args.limit)
    print(f"  {len(split_interactions):,} {args.split} impressions to featurize")

    category_lookup = dict(zip(articles["article_id"], articles["category"]))
    # Computed ONCE over the whole (all-splits) interactions table -- see
    # pipeline.features_common.compute_first_seen_times's docstring for why
    # this is safe to do without per-split gating (the leakage guard is a
    # per-row point-in-time check applied downstream, not a coarse
    # split-level restriction on this computation itself).
    first_seen = compute_first_seen_times(interactions)

    if args.dataset == "mind":
        subcategory_lookup = dict(zip(articles["article_id"], articles["subcategory"]))
        build_fn = lambda chunk: build_mind_candidate_features(
            chunk, category_lookup, subcategory_lookup, article_pop, first_seen, semantic,
            recent_n, recency_decay,
        )
    else:
        ebnerd_dir = os.path.join(args.data_dir, args.ebnerd_bundle)
        raw_articles = load_ebnerd_articles_raw(ebnerd_dir)
        published_time_lookup = dict(zip(
            raw_articles["article_id"].astype(str),
            pd.to_datetime(raw_articles["published_time"], errors="coerce"),
        ))
        print("  precomputing session/engagement history (train+val, strictly-prior-only) ...")
        session_engagement = precompute_session_engagement(ebnerd_dir)
        build_fn = lambda chunk: build_ebnerd_candidate_features(
            chunk, category_lookup, published_time_lookup, article_pop, first_seen,
            session_engagement, semantic, recent_n, recency_decay,
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"candidate_features_{args.split}.parquet")
    writer = None
    n_written = 0
    n_chunks = (len(split_interactions) + args.chunk_size - 1) // args.chunk_size or 1
    for i in range(0, len(split_interactions), args.chunk_size):
        chunk = split_interactions.iloc[i:i + args.chunk_size]
        feats = build_fn(chunk)
        if feats.empty:
            continue
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n_written += len(feats)
        print(f"  chunk {i // args.chunk_size + 1}/{n_chunks}: "
              f"{len(chunk):,} impressions -> {len(feats):,} candidate rows "
              f"(total written: {n_written:,})")
    if writer is not None:
        writer.close()

    print(f"\nWrote {n_written:,} candidate-level feature rows -> {out_path}")


if __name__ == "__main__":
    main()
