"""Shared data-loading/feature-joining mechanics for the A2 re-ranker,
used by both scripts/train_reranker.py and scripts/evaluate_reranker.py --
moved here (rather than staying private to train_reranker.py) once a
second script needed the exact same "load Q1 features, join Assignment-1
retrieval scores, return one clean DataFrame" logic; duplicating it would
have risked the two scripts' feature tables silently drifting apart.

FEATURE_COLUMNS stays deliberately dataset-specific (MIND vs EB-NeRD have
genuinely different Q1 feature sets, see pipeline/features_{mind,ebnerd}.py)
-- only the retrieval-score columns (bm25_score/semantic_score/fusion_score)
and the loading/joining mechanics below are shared.
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.reranker_scores import compute_retrieval_scores

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
    hyperparameters, or to evaluate rather than train) reuses the persisted
    file instead of recomputing it -- `cache_path` itself encodes
    recent_n/recency_decay/fusion_alpha (see caller) so a changed
    hyperparameter can't silently reuse a stale one."""
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
        # see scripts/train_reranker.py's module docstring. A fixed seed
        # keeps this reproducible.
        rng = np.random.default_rng(seed)
        all_ids = feats["impression_id"].unique()
        if sample_size < len(all_ids):
            keep_ids = set(rng.choice(all_ids, size=sample_size, replace=False))
            feats = feats[feats["impression_id"].isin(keep_ids)]
            score_table = score_table[score_table["impression_id"].isin(keep_ids)]
            print(f"  sampled down to {sample_size:,}/{len(all_ids):,} {split} impressions "
                  f"(seed={seed})")

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
        # float32 halves the numeric footprint of the columns actually
        # trained/scored on vs. pandas' float64 default -- GBDTs don't need
        # float64 precision, and at tens-of-millions-of-rows scale this is
        # a real, non-cosmetic memory saving, not premature optimization.
        for col in feature_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype("float32")

    return merged.sort_values("impression_id", kind="stable").reset_index(drop=True)
