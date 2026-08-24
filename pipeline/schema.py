"""Column-name constants for the unified schema that both MIND and EB-NeRD get
mapped into. Keeping these in one place is what lets the rest of the pipeline
(split, feature store) stay dataset-agnostic.
"""

# -- unified articles table -------------------------------------------------
# one row per article, no split column (articles are shared across splits;
# split-dependent stats like popularity live in the feature store, not here).
ARTICLE_COLUMNS = [
    "dataset",         # "mind" | "ebnerd"
    "article_id",      # str, native id, unique within a dataset
    "title",
    "abstract",         # MIND: abstract. EB-NeRD: subtitle (closest analogue).
    "body",             # MIND: always None (not shipped). EB-NeRD: body.
    "category",
    "subcategory",       # MIND: subcategory. EB-NeRD: None (no equivalent field).
    "entities",          # list[str] of entity ids/labels, dataset-specific source.
    "url",
    "published_time",    # MIND: None (not shipped). EB-NeRD: published_time.
]

# -- unified interactions (impression-level behaviors) table ----------------
INTERACTION_COLUMNS = [
    "dataset",
    "impression_id",
    "user_id",
    "impression_time",
    "history_article_ids",    # list[str], user's click history as of this impression
    "candidate_article_ids",  # list[str], articles shown in this impression
    "labels",                 # list[int] (0/1), aligned with candidate_article_ids
    "split",                  # "train" | "val" | "test", set by pipeline.split
]

# -- unified click-history (long format) table -------------------------------
# One row per (user, clicked article, timestamp). Built from *timestamped*
# clicks only, so it can be safely truncated at a split cutoff for recency
# features without leaking future clicks. See pipeline/mind.py and
# pipeline/ebnerd.py docstrings for how each dataset's timestamps are sourced.
CLICK_HISTORY_COLUMNS = [
    "dataset",
    "user_id",
    "article_id",
    "click_time",
    "split",   # which split this click event itself falls into (diagnostic only)
]

SPLITS = ("train", "val", "test")
