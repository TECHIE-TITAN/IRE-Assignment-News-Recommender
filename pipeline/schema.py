"""Column-name constants for the unified MIND schema.

Keeping these in one place is what lets the rest of the pipeline (split,
feature store, retrieval) stay independent of MIND's raw TSV layout.
"""

# -- unified articles table -------------------------------------------------
# one row per article, no split column (articles from train+dev are pooled;
# split-dependent stats like popularity live in the feature store, not here).
ARTICLE_COLUMNS = [
    "article_id",   # str, native news_id, unique
    "title",
    "abstract",
    "category",
    "subcategory",
    "entities",     # list[str] of deduped Wikidata IDs (title + abstract entities)
    "url",
]

# -- unified interactions (impression-level behaviors) table ----------------
INTERACTION_COLUMNS = [
    "impression_id",
    "user_id",
    "impression_time",
    "history_article_ids",    # list[str], user's click history as of this impression
    "candidate_article_ids",  # list[str], articles shown in this impression
    "labels",                 # list[int|None] (0/1), aligned with candidate_article_ids;
                               # None entries for the unlabeled MINDlarge_test split
    "split",                  # "train" | "val" | "test"
]

# -- unified click-history (long format) table -------------------------------
# One row per (user, clicked article, timestamp). Built from *timestamped*
# clicks only (positive-labelled impression candidates), so it can be safely
# truncated at a split cutoff for recency features without leaking future
# clicks. See pipeline/mind.py docstring for why MIND has no other source of
# per-click timestamps.
CLICK_HISTORY_COLUMNS = [
    "user_id",
    "article_id",
    "click_time",
    "split",   # which split this click event itself falls into (diagnostic only)
]

SPLITS = ("train", "val", "test")
