"""A2 Q1 candidate-level behavioural features for MIND.

MIND ships no session id, no dwell time, and no article publish timestamp
(pipeline/schema.py's ARTICLE_COLUMNS has no `published_time` -- see its
own comment on why) -- so "session features" and true "freshness" are
structurally unavailable here, not omitted by oversight. What MIND DOES
give us in its favor: `history_article_ids` is the dataset's own literal,
already-leakage-safe as-of-this-impression history (no join needed), and
`candidate_article_ids` preserves real display order (position bias is
free). Freshness is approximated by "days since first seen in our own
logs" -- see pipeline.features_common.compute_first_seen_times/
days_since_first_seen for exactly how that stays leakage-safe, since this
proxy COULD leak future information if computed carelessly.

Feature columns produced (see `build_mind_candidate_features`):
    candidate_position          -- 1-indexed display position
    history_length              -- len(history_article_ids), uncapped
    history_category_match      -- recency-weighted category-match score
    history_subcategory_match   -- same, for subcategory
    history_embedding_sim       -- cosine sim vs recency-weighted mean-pooled
                                    history embedding (Assignment 1's semantic index)
    article_train_ctr           -- train-only click-through rate (Assignment 1 feature, reused)
    article_train_impressions   -- train-only exposure count (Assignment 1 feature, reused)
    article_days_since_first_seen -- freshness proxy; see
                                      pipeline.features_common.compute_first_seen_times/
                                      days_since_first_seen for exactly how this stays
                                      leakage-safe (a per-row, point-in-time check, not a
                                      coarse split-level one -- the latter looks safe for
                                      val/test but isn't safe for a split's OWN internal
                                      chronology, e.g. featurizing train itself)
"""

import pandas as pd

from pipeline.features_common import (
    days_since_first_seen,
    history_embedding_similarity_map,
    position_bias_map,
    weighted_category_match_map,
)


def build_mind_candidate_features(interactions_chunk, category_lookup, subcategory_lookup,
                                    article_pop, first_seen_times, semantic_index,
                                    recent_n=20, recency_decay=0.85):
    """One row per (impression_id, candidate_article_id) for every
    impression in `interactions_chunk`. `article_pop`: DataFrame indexed by
    article_id with `train_ctr`/`train_impressions` (from
    pipeline.feature_store.build_article_features, reused as-is -- already
    train-only leakage-safe). `first_seen_times`: a Series from
    `pipeline.features_common.compute_first_seen_times`, computed by the
    caller ONCE over the whole (all-splits) interactions table -- see that
    function's docstring for why no split-scoping is needed here; the
    leakage guard lives in `days_since_first_seen`'s point-in-time gate
    instead."""
    rows = []
    for imp in interactions_chunk.itertuples(index=False):
        history = imp.history_article_ids
        candidates = imp.candidate_article_ids
        if len(candidates) == 0:
            continue

        pos_map = position_bias_map(candidates)
        cat_map = weighted_category_match_map(history, category_lookup, recent_n, recency_decay)
        subcat_map = weighted_category_match_map(history, subcategory_lookup, recent_n, recency_decay)
        emb_map = history_embedding_similarity_map(history, semantic_index, candidates, recent_n, recency_decay)
        cand_cats = [category_lookup.get(c) for c in candidates]
        cand_subcats = [subcategory_lookup.get(c) for c in candidates]

        for i, cid in enumerate(candidates):
            rows.append((
                imp.impression_id, imp.user_id, cid,
                imp.labels[i] if imp.labels is not None and len(imp.labels) > i else None,
                imp.split,
                pos_map[cid],
                len(history),
                cat_map.get(cand_cats[i], 0.0),
                subcat_map.get(cand_subcats[i], 0.0),
                float(emb_map.get(cid, 0.0)),
                cid,  # kept for the two vectorized .map() joins below
            ))

    cols = ["impression_id", "user_id", "candidate_article_id", "label", "split",
            "candidate_position", "history_length", "history_category_match",
            "history_subcategory_match", "history_embedding_sim", "_article_id"]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out.drop(columns=["_article_id"])

    out["article_train_ctr"] = out["_article_id"].map(article_pop["train_ctr"]).fillna(0.0)
    out["article_train_impressions"] = out["_article_id"].map(article_pop["train_impressions"]).fillna(0)

    impression_time = interactions_chunk.set_index("impression_id")["impression_time"]
    ref_time = out["impression_id"].map(impression_time)
    out["article_days_since_first_seen"] = days_since_first_seen(out["_article_id"], ref_time, first_seen_times)

    return out.drop(columns=["_article_id"])
