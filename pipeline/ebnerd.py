"""Load EB-NeRD (ebnerd_small for now) raw parquet files and map them into
the unified schema defined in pipeline/schema.py -- the same schema
pipeline/mind.py maps MIND into, so every downstream script (retrieval/,
scripts/build_indices.py, scripts/evaluate_*.py) works against EB-NeRD
without modification once this loader hands it a conforming
articles/interactions table.

EB-NeRD quirks that shape this loader:
  - `articles.parquet` has a `body` field and both a numeric `category`
    code and a human-readable `category_str`; the unified schema keeps
    only what retrieval/eval actually use (title/abstract/entities/
    category/subcategory/url) -- `body` and `published_time` are dropped,
    not carried as unused columns, since nothing downstream reads them.
    `subtitle` is EB-NeRD's closest analogue to MIND's `abstract`.
  - EB-NeRD ships no entity *embeddings* (unlike MIND's TransE vectors) --
    `ner_clusters`/`entity_groups` are named-entity surface strings (e.g.
    "Willy Strube"), not Wikidata IDs, so they're carried into `entities`
    for schema parity but can't be used for embedding fusion
    (retrieval/entity_embeddings.py's fusion path is simply never invoked
    for this dataset -- fit_indices()'s entity_vectors stays None).
  - `history.parquet` (one row per user per split) carries per-click
    timestamps (`impression_time_fixed`, parallel to `article_id_fixed`),
    so the unified click-history table is built directly from it, unlike
    MIND's, which has to be derived from positive-labelled impressions.
  - `behaviors.parquet` doesn't embed each impression's history inline
    (unlike MIND) -- it's joined from history.parquet by user_id, scoped
    per split (train impressions join train/history.parquet, validation
    impressions join validation/history.parquet).
  - Scope: this loader only covers train+validation (ebnerd_small), matching
    Q1's literal spec ("MIND-small and EB-NeRD demo/small"). The large,
    unlabeled test set (ebnerd_testset, 13.5M impressions) is handled
    separately and directly by scripts/generate_predictions_ebnerd.py,
    streamed from raw parquet rather than folded into this pipeline's
    interactions.parquet -- unlike MIND's build_pipeline.py, which does
    fold MINDlarge_test in (harmless there at 2.37M rows, but EB-NeRD's
    test set is ~5.7x larger and needs a per-impression history join that's
    expensive at that scale, so it's deliberately excluded here).
"""

import os

import numpy as np
import pandas as pd

from pipeline.schema import ARTICLE_COLUMNS, INTERACTION_COLUMNS

SPLIT_DIRS = {"train": "train", "val": "validation"}


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    return [x]


def load_articles(ebnerd_dir):
    return pd.read_parquet(os.path.join(ebnerd_dir, "articles.parquet"))


def load_history(split_dir):
    return pd.read_parquet(os.path.join(split_dir, "history.parquet"))


def load_behaviors(split_dir):
    df = pd.read_parquet(os.path.join(split_dir, "behaviors.parquet"))
    if "impression_time" in df.columns:
        df["impression_time"] = pd.to_datetime(df["impression_time"], errors="coerce")
    return df


def articles_from_raw_df(articles):
    """Maps a raw articles.parquet-shaped dataframe to the unified article
    schema. Used both for ebnerd_small (train+val) and, by
    scripts/generate_predictions_ebnerd.py, for ebnerd_testset's own
    articles.parquet."""
    # `ner_clusters` is the actual named-entity mention strings (e.g.
    # "Willy Strube"); `entity_groups` is the parallel per-mention type
    # label (e.g. "PER"/"ORG"/"LOC"), not itself an entity -- deliberately
    # not folded into `entities` here, unlike an earlier version of this
    # loader that unioned both into one list and ended up with mixed
    # names+type-labels in the same field.
    entities = (articles["ner_clusters"].apply(lambda x: [str(e) for e in _to_list(x)])
                if "ner_clusters" in articles else pd.Series([[]] * len(articles)))

    subcat = (articles["subcategory"].apply(lambda x: ",".join(str(c) for c in _to_list(x)))
              if "subcategory" in articles else "")
    category = articles["category_str"] if "category_str" in articles else articles.get("category")

    out = pd.DataFrame({
        "article_id": articles["article_id"].astype(str),
        "title": articles["title"],
        "abstract": articles["subtitle"] if "subtitle" in articles else None,
        "category": category,
        "subcategory": subcat,
        "entities": entities,
        "url": articles["url"] if "url" in articles else None,
    })
    return out[ARTICLE_COLUMNS]


def load_ebnerd_articles(ebnerd_dir):
    return articles_from_raw_df(load_articles(ebnerd_dir))


def _build_history_lookup(history_df):
    """user_id -> list[str] of article ids (chronological, matching how
    MIND's inline `history` field is consumed elsewhere)."""
    id_col = "article_id_fixed"
    ids_series = history_df[id_col] if id_col in history_df.columns else pd.Series([[]] * len(history_df))
    lookup = {}
    for uid, aids in zip(history_df["user_id"], ids_series):
        lookup[uid] = [str(a) for a in _to_list(aids)]
    return lookup


def load_ebnerd_interactions(ebnerd_dir):
    """Concatenation of train + validation behaviors.parquet (ebnerd_small
    only -- see module docstring), each row's history joined in from that
    split's history.parquet, mapped to the unified interactions schema.
    `split` is set directly from which directory a row came from (train ->
    "train", validation -> "val"), matching MIND's file-is-the-split
    convention -- see pipeline/split.py's boundary check, which this
    dataset also passes (train ends 2023-05-25 07:00, validation starts
    the same instant, contiguous with no overlap)."""
    frames = []
    for split, dirname in SPLIT_DIRS.items():
        split_dir = os.path.join(ebnerd_dir, dirname)
        beh = load_behaviors(split_dir)
        hist_lookup = _build_history_lookup(load_history(split_dir))

        history_article_ids = beh["user_id"].map(lambda uid: hist_lookup.get(uid, []))
        inview = beh["article_ids_inview"].apply(lambda x: [str(a) for a in _to_list(x)])
        clicked = beh["article_ids_clicked"].apply(lambda x: {str(a) for a in _to_list(x)})
        labels = [
            [1 if aid in clicked_set else 0 for aid in cand]
            for cand, clicked_set in zip(inview, clicked)
        ]

        frames.append(pd.DataFrame({
            "impression_id": f"{split}_" + beh["impression_id"].astype(str),
            "user_id": beh["user_id"],
            "impression_time": beh["impression_time"],
            "history_article_ids": history_article_ids,
            "candidate_article_ids": inview,
            "labels": labels,
            "split": split,
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[INTERACTION_COLUMNS]


def derive_ebnerd_click_history(ebnerd_dir):
    """Built directly from history.parquet's (article_id_fixed,
    impression_time_fixed) pairs, which are EB-NeRD's authoritative
    timestamped click log -- unlike MIND, no need to derive it from
    positive-labelled impressions. Unioned across train + validation and
    deduplicated, since a user's validation-split history snapshot is a
    superset of their train-split one."""
    frames = []
    for split, dirname in SPLIT_DIRS.items():
        hist = load_history(os.path.join(ebnerd_dir, dirname))
        df = pd.DataFrame({
            "user_id": hist["user_id"],
            "article_id_fixed": hist["article_id_fixed"].apply(_to_list),
            "impression_time_fixed": hist["impression_time_fixed"].apply(_to_list),
        })
        df = df.explode(["article_id_fixed", "impression_time_fixed"])
        df = df.dropna(subset=["article_id_fixed"])
        df["split"] = split
        frames.append(df.rename(columns={
            "article_id_fixed": "article_id", "impression_time_fixed": "click_time",
        }))

    ch = pd.concat(frames, ignore_index=True)[["user_id", "article_id", "click_time", "split"]]
    ch["article_id"] = ch["article_id"].astype(str)
    ch["click_time"] = pd.to_datetime(ch["click_time"], errors="coerce")
    ch = ch.drop_duplicates(subset=["user_id", "article_id", "click_time"])
    return ch
