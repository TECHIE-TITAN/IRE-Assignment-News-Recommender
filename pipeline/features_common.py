"""Shared, dataset-agnostic feature-engineering math for Assignment 2 Q1.

Deliberately small: the actual per-dataset feature *sets* live in
pipeline/features_mind.py and pipeline/features_ebnerd.py and are NOT
unified into one schema -- MIND and EB-NeRD have genuinely different
available signals (EB-NeRD has session_id/read_time/scroll_percentage and
a true published_time; MIND has neither), so forcing one shared feature
table would mean either faking columns MIND doesn't have data for, or
throwing away signal EB-NeRD does have. This module only holds the
handful of computations that are identical, dataset-agnostic math once
given the right inputs (a history_article_ids list, a category lookup, a
fitted Assignment-1 semantic index) -- mirroring how retrieval/text_utils.py
is shared infra for BM25/LSA query construction without BM25 vs LSA
themselves being unified.

Every function here operates on ONE impression's already-leakage-safe
history (`history_article_ids` is each dataset's own as-of-this-impression
field -- see pipeline/mind.py / pipeline/ebnerd.py's derivation
docstrings -- so no additional cutoff filtering is needed inside this
module for history-based features; the dataset-specific modules handle
leakage safety for the signals that AREN'T pre-filtered this way, e.g.
EB-NeRD's session/engagement history and article popularity/freshness).
"""

import pandas as pd

from retrieval.lsa import mean_pool_user_vector
from retrieval.text_utils import recency_weights


def recent_history_with_weights(history_ids, recent_n=20, decay=0.85):
    """Last `recent_n` of `history_ids` (chronologically ascending, so the
    *last* entry is the most recent click -- both datasets' history field
    uses this convention, see retrieval/text_utils.weighted_query_terms)
    with their recency weights. Returns `([], [])` for a cold-start
    (empty-history) impression -- callers must handle that, not treat it
    as an error."""
    if len(history_ids) == 0:
        return [], []
    recent = list(history_ids)[-recent_n:]
    return recent, recency_weights(len(recent), decay)


def weighted_category_match_map(history_ids, category_lookup, recent_n=20, decay=0.85):
    """Recency-weighted distribution over the categories in the user's
    recent history, normalized to sum to 1 across categories actually seen
    -- so a candidate's match score is comparable regardless of history
    length or how much of the decay-weighted mass landed on articles
    missing a category. Computed ONCE per impression (O(recent_n)); a
    candidate's feature value is then a cheap `dist.get(category, 0.0)`
    lookup after exploding, not recomputed per candidate."""
    recent, weights = recent_history_with_weights(history_ids, recent_n, decay)
    if not recent:
        return {}
    dist = {}
    total = 0.0
    for aid, w in zip(recent, weights):
        cat = category_lookup.get(aid)
        if cat is not None:
            dist[cat] = dist.get(cat, 0.0) + w
            total += w
    if total == 0.0:
        return {}
    return {c: v / total for c, v in dist.items()}


def history_embedding_similarity_map(history_ids, semantic_index, candidate_ids, recent_n=20, decay=0.85):
    """Cosine similarity between the recency-weighted mean-pooled embedding
    of the user's recent history and each candidate -- the "click-history
    ... embeddings" feature A2 Q1 asks for, built directly on Assignment
    1's already-fitted, already-persisted semantic index (LSA or SBERT,
    whichever this run was built with -- both share the same
    get_embedding/score_candidates surface, see retrieval/build_indices.py)
    rather than introducing a new embedding step. `mean_pool_user_vector`
    and `score_candidates` both already degrade gracefully (return
    None / all-zero) for a cold-start or fully-out-of-vocabulary history,
    so a zero similarity -- not a fabricated guess -- is what a cold-start
    impression correctly gets here."""
    recent, weights = recent_history_with_weights(history_ids, recent_n, decay)
    if not recent:
        return {cid: 0.0 for cid in candidate_ids}
    embeddings = [semantic_index.get_embedding(aid) for aid in recent]
    user_vec = mean_pool_user_vector(embeddings, weights=weights)
    scores = semantic_index.score_candidates(user_vec, candidate_ids)
    return dict(zip(candidate_ids, scores))


def position_bias_map(candidate_ids):
    """1-indexed display position of each candidate within its own
    impression's own candidate list -- both datasets preserve the
    dataset's own display order in this field (MIND's impression string,
    EB-NeRD's article_ids_inview), so no re-derivation needed."""
    return {cid: i + 1 for i, cid in enumerate(candidate_ids)}


def compute_first_seen_times(interactions):
    """Per-article earliest `impression_time` at which it appears as a
    candidate, across the ENTIRE interactions table (all splits pooled) --
    a fixed fact about each article, the same way a true publish date would
    be. This is deliberately NOT split-scoped on its own: computing it once
    over the whole table and then gating its USE per-impression (see
    `days_since_first_seen`) is both simpler and safer than restricting the
    input to "prior splits only", which looks safe for val/test (train
    always fully precedes them) but is NOT safe for a split's own internal
    ordering -- an early train impression could otherwise pick up an
    article's first-appearance time from later in that same train split. A
    per-impression, point-in-time comparison against the impression's own
    time (below) closes that gap uniformly for every split, including a
    split's own internal chronology, with no special-casing needed."""
    exploded = interactions[["impression_time", "candidate_article_ids"]].explode("candidate_article_ids")
    exploded = exploded.rename(columns={"candidate_article_ids": "article_id"}).dropna(subset=["article_id"])
    return exploded.groupby("article_id")["impression_time"].min()


def days_since_first_seen(article_ids, impression_times, first_seen_times):
    """Vectorized freshness-proxy computation. `article_ids`/`impression_times`:
    parallel Series, one per candidate row. `first_seen_times`: a
    Series OR plain dict of {article_id: Timestamp}. Any row where the
    article's global first-seen time is AT OR AFTER that row's own
    impression_time (including "never seen before this row at all," i.e.
    NaT/missing) gets the neutral max-observed-age value instead of a real
    day count -- 0 would falsely claim "brand new," and a negative number
    would be nonsensical. This is the actual leakage guard: it's what
    stops a future first-appearance from ever contributing a real (small,
    "fresh-looking") day count to a row that predates it.

    `pd.to_datetime(...)` wraps the `.map()` result deliberately: mapping
    against a plain dict (as opposed to a Series) doesn't reliably infer
    datetime64 dtype when some article_ids are missing from the dict --
    it can come back as `object` dtype (Timestamps mixed with plain NaN),
    which silently breaks the `.dt` accessor below. Forcing it back to
    datetime64 explicitly costs nothing when the input was already correct
    (a no-op) and removes any doubt when it wasn't."""
    first_seen = pd.to_datetime(article_ids.map(first_seen_times))
    days = (impression_times - first_seen).dt.total_seconds() / 86400.0
    valid = days >= 0
    max_age = days[valid].max() if valid.any() else 0.0
    return days.where(valid, max_age)
