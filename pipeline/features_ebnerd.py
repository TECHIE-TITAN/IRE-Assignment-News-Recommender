"""A2 Q1 candidate-level behavioural features for EB-NeRD.

EB-NeRD's raw behaviors.parquet has real session/engagement columns MIND
has no equivalent of at all: `session_id`, `read_time`, `scroll_percentage`
(confirmed against the actual bundle -- see data/ebnerd_small/train/
behaviors.parquet's columns). It also has a true article `published_time`
(data/ebnerd_small/articles.parquet), unlike MIND. This module is
deliberately separate from pipeline/features_mind.py rather than sharing
one schema, precisely so these dataset-specific signals aren't diluted
into least-common-denominator columns.

A critical anti-gaming trap specific to this data (Q9): `read_time` /
`scroll_percentage` / `next_read_time` / `next_scroll_percentage` on a
behaviors row describe how the user engaged with what THAT impression
showed them -- they are the OUTCOME of the impression being scored, not
information available before it happened. Using a row's own read_time as
a feature for scoring that same row would be leakage in the most direct
sense (using the answer to predict the answer). The features built here
instead use ONLY strictly-prior engagement: `user_avg_past_read_time` /
`user_avg_past_scroll_pct` (that user's own history, before this
impression) and `session_prior_click_count` (earlier impressions in the
same session, before this one) -- see `precompute_session_engagement`.

Only `category` (not `subcategory`) is used for the category-match
feature: EB-NeRD's unified `subcategory` column is a comma-joined string of
multiple numeric IDs (e.g. `"433,434"`, or empty) -- confirmed against the
actual data -- not a single scalar category like MIND's, so an exact-match
feature on it would rarely fire even when two articles genuinely share a
subcategory. Splitting/exploding it into a real multi-label match is a
reasonable future improvement, not implemented here.

Feature columns produced (see `build_ebnerd_candidate_features`):
    candidate_position           -- 1-indexed display position
    history_length                -- len(history_article_ids)
    history_category_match        -- recency-weighted category-match score
    history_embedding_sim         -- cosine sim vs recency-weighted mean-pooled
                                      history embedding (Assignment 1's semantic index)
    article_train_ctr             -- train-only click-through rate
    article_train_impressions     -- train-only exposure count
    article_days_since_publish    -- TRUE freshness (EB-NeRD ships published_time)
    session_prior_click_count     -- clicks in THIS session strictly before this impression
    user_avg_past_read_time       -- that user's own mean read_time on strictly-prior impressions
    user_avg_past_scroll_pct      -- same, for scroll_percentage
"""

import os

import pandas as pd

from pipeline.ebnerd import SPLIT_DIRS, _to_list, load_behaviors
from pipeline.features_common import (
    days_since_first_seen,
    history_embedding_similarity_map,
    position_bias_map,
    weighted_category_match_map,
)


def _expanding_mean_of_strictly_prior(values, group_keys):
    """Per-group expanding mean of `values`, using only STRICTLY PRIOR rows
    (never the current row's own value) -- `values`/`group_keys` must
    already be sorted by (group, time).

    Deliberately built from `shift`/`cumsum`/`notna` (unambiguous groupby
    TRANSFORMS, guaranteed index-aligned to the input by the pandas API)
    rather than `groupby(...).apply(lambda s: s.shift(1).expanding().mean())`
    -- that pattern was tried first and produced a real, silent misalignment
    bug (caught by tests/test_feature_leakage.py's
    test_user_engagement_average_is_per_user_not_global): `.apply()`'s
    result combines per-group in GROUP-KEY order, and the `.reset_index(drop=True)`
    needed to strip its index then builds a fresh RangeIndex over THAT
    grouped order -- which silently misaligns against the caller's index
    (still in sort_values' original-label order) once assigned back,
    because DataFrame column assignment aligns by index LABEL, not
    position. `shift`/`cumsum` never go through `.apply()`, so this
    ambiguity doesn't exist for them."""
    shifted = values.groupby(group_keys).shift(1)
    prior_sum = shifted.groupby(group_keys).cumsum()
    prior_count = shifted.notna().groupby(group_keys).cumsum()
    # prior_count==0 implies prior_sum==0 too (nothing to cumsum yet), so
    # plain division already yields NaN there (0.0/0.0), not a divide-by-a-
    # real-zero case -- no separate zero-guard needed.
    return prior_sum / prior_count


def _session_engagement_from_frame(all_beh):
    """Pure computation half of `precompute_session_engagement`, split out
    so it's unit-testable on a small synthetic DataFrame without needing
    real parquet files on disk (see tests/test_feature_leakage.py) --
    `all_beh` needs columns impression_id/user_id/session_id/impression_time/
    read_time/scroll_percentage/n_clicked, already unioned across splits.

    Leakage-safety, precisely: `.shift(1)` (session click count) and
    `_expanding_mean_of_strictly_prior` (engagement averages) both exclude
    the CURRENT row's own value from its own feature -- only strictly-
    earlier rows (by impression_time, within the same session or same
    user) ever contribute."""
    all_beh = all_beh.sort_values(["session_id", "impression_time"])
    all_beh["session_prior_click_count"] = (
        all_beh.groupby("session_id")["n_clicked"].cumsum().sub(all_beh["n_clicked"])
    )

    all_beh = all_beh.sort_values(["user_id", "impression_time"])
    all_beh["user_avg_past_read_time"] = _expanding_mean_of_strictly_prior(all_beh["read_time"], all_beh["user_id"])
    all_beh["user_avg_past_scroll_pct"] = _expanding_mean_of_strictly_prior(all_beh["scroll_percentage"], all_beh["user_id"])

    return all_beh.set_index("impression_id")[
        ["session_prior_click_count", "user_avg_past_read_time", "user_avg_past_scroll_pct"]
    ]


def precompute_session_engagement(ebnerd_dir):
    """Loads train+validation behaviors.parquet (small enough to hold in
    memory in full -- 477K rows for ebnerd_small, unlike MIND's raw
    behaviors.tsv which needs streaming), unions them, and delegates to
    `_session_engagement_from_frame`. Returns a DataFrame indexed by the
    SAME `impression_id` convention as pipeline.ebnerd.load_ebnerd_interactions
    (`f"{split}_{original_impression_id}"`), so the caller can join it in
    with a plain `.map()`. Sorting train+validation together (not per-split)
    is deliberate: a user's validation-period impression should see their
    train-period session/engagement history too, exactly matching how
    pipeline.ebnerd.derive_ebnerd_click_history already unions history
    across splits for the same reason."""
    frames = []
    for split, dirname in SPLIT_DIRS.items():
        beh = load_behaviors(os.path.join(ebnerd_dir, dirname))
        frames.append(pd.DataFrame({
            "impression_id": f"{split}_" + beh["impression_id"].astype(str),
            "user_id": beh["user_id"],
            "session_id": beh["session_id"],
            "impression_time": beh["impression_time"],
            "read_time": beh["read_time"],
            "scroll_percentage": beh["scroll_percentage"],
            "n_clicked": beh["article_ids_clicked"].apply(lambda x: len(_to_list(x))),
        }))
    return _session_engagement_from_frame(pd.concat(frames, ignore_index=True))


def build_ebnerd_candidate_features(interactions_chunk, category_lookup, published_time_lookup,
                                      article_pop, first_seen_times, session_engagement, semantic_index,
                                      recent_n=20, recency_decay=0.85):
    """One row per (impression_id, candidate_article_id). `published_time_lookup`:
    {article_id: Timestamp} from the raw articles table. `first_seen_times`:
    from `pipeline.features_common.compute_first_seen_times`, called ONCE by
    the caller over the whole interactions table (not per-split -- see that
    function's docstring for why) -- used only as a fallback for the small
    fraction of articles with a null `published_time`. `session_engagement`:
    the DataFrame returned by `precompute_session_engagement`, joined in by
    `impression_id`."""
    rows = []
    for imp in interactions_chunk.itertuples(index=False):
        history = imp.history_article_ids
        candidates = imp.candidate_article_ids
        if len(candidates) == 0:
            continue

        pos_map = position_bias_map(candidates)
        cat_map = weighted_category_match_map(history, category_lookup, recent_n, recency_decay)
        emb_map = history_embedding_similarity_map(history, semantic_index, candidates, recent_n, recency_decay)
        cand_cats = [category_lookup.get(c) for c in candidates]

        for i, cid in enumerate(candidates):
            rows.append((
                imp.impression_id, imp.user_id, cid,
                imp.labels[i] if imp.labels is not None and len(imp.labels) > i else None,
                imp.split,
                pos_map[cid],
                len(history),
                cat_map.get(cand_cats[i], 0.0),
                float(emb_map.get(cid, 0.0)),
                cid,
            ))

    cols = ["impression_id", "user_id", "candidate_article_id", "label", "split",
            "candidate_position", "history_length", "history_category_match",
            "history_embedding_sim", "_article_id"]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out.drop(columns=["_article_id"])

    out["article_train_ctr"] = out["_article_id"].map(article_pop["train_ctr"]).fillna(0.0)
    out["article_train_impressions"] = out["_article_id"].map(article_pop["train_impressions"]).fillna(0)

    impression_time = interactions_chunk.set_index("impression_id")["impression_time"]
    ref_time = out["impression_id"].map(impression_time)

    # Per-article reference time = published_time where known, else the
    # first-seen-in-logs proxy -- combined BEFORE the point-in-time gate
    # (days_since_first_seen) runs, so a corrupted/future published_time
    # gets the same "not yet seen" fallback treatment as a too-late
    # first-seen time would, not a separate, unguarded code path.
    #
    # `days_since_first_seen` expects an article-id-indexed lookup (it does
    # its own `.map()` internally), so the fallback merge has to stay
    # article-indexed too -- done as a plain dict merge rather than
    # `pd.Series(published_time_lookup).combine_first(first_seen_times)`:
    # those two Series only partially overlap in which articles they cover
    # (published_time_lookup spans every EB-NeRD article, first_seen_times
    # only ones that appeared as a candidate), and combine_first's
    # index-union path throws a pandas FutureWarning for that shape on
    # datetime64 data (a real, if currently-harmless, signal that its
    # internal behavior will change). A dict merge has no such concern.
    reference_time_lookup = first_seen_times.to_dict()
    reference_time_lookup.update({k: v for k, v in published_time_lookup.items() if pd.notna(v)})
    out["article_days_since_publish"] = days_since_first_seen(out["_article_id"], ref_time, reference_time_lookup)

    for col in ["session_prior_click_count", "user_avg_past_read_time", "user_avg_past_scroll_pct"]:
        out[col] = out["impression_id"].map(session_engagement[col]).fillna(0.0)

    return out.drop(columns=["_article_id"])
