"""Load EB-NeRD (ebnerd_demo / ebnerd_small) raw files and map them into the
unified schema defined in pipeline/schema.py.

EB-NeRD quirks that shape this loader:
  - `articles.parquet` has a `body` field (MIND doesn't); `subtitle` is the
    closest analogue to MIND's `abstract`.
  - `history.parquet` (one row per user per split) DOES carry per-click
    timestamps (`impression_time_fixed`, parallel to `article_id_fixed`), so
    the unified click-history table is built directly from it rather than
    derived from positive-labelled impressions the way MIND's is.
  - `behaviors.parquet` doesn't embed each impression's history inline (unlike
    MIND) — it must be joined from history.parquet by user_id; the join is
    split-scoped (train impressions join train/history.parquet, validation
    impressions join validation/history.parquet), matching how EB-NeRD itself
    scopes each history snapshot to the impressions it precedes.

Column names follow the official EB-NeRD release schema. A few optional
columns are looked up defensively (present/absent checked) in case a given
bundle (demo vs. small) omits one.
"""

import os

import numpy as np
import pandas as pd

from pipeline.schema import ARTICLE_COLUMNS, INTERACTION_COLUMNS


def _col(df, *candidates):
    """Returns the first candidate column name present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def load_articles(data_dir):
    return pd.read_parquet(os.path.join(data_dir, "articles.parquet"))


def load_history(split_dir):
    return pd.read_parquet(os.path.join(split_dir, "history.parquet"))


def load_behaviors(split_dir):
    df = pd.read_parquet(os.path.join(split_dir, "behaviors.parquet"))
    if "impression_time" in df.columns:
        df["impression_time"] = pd.to_datetime(df["impression_time"], errors="coerce")
    return df


def load_ebnerd_articles(data_dir):
    articles = load_articles(data_dir)

    title_col = _col(articles, "title")
    subtitle_col = _col(articles, "subtitle")
    body_col = _col(articles, "body")
    cat_col = _col(articles, "category_str", "category")
    subcat_col = _col(articles, "subcategory")
    entity_col = _col(articles, "ner_clusters", "entity_groups")
    url_col = _col(articles, "url")
    pub_col = _col(articles, "published_time")

    out = pd.DataFrame({
        "dataset": "ebnerd",
        "article_id": articles["article_id"].astype(str),
        "title": articles[title_col] if title_col else None,
        "abstract": articles[subtitle_col] if subtitle_col else None,
        "body": articles[body_col] if body_col else None,
        "category": articles[cat_col] if cat_col else None,
        "subcategory": articles[subcat_col].apply(_to_list) if subcat_col else None,
        "entities": articles[entity_col].apply(_to_list) if entity_col else [[]] * len(articles),
        "url": articles[url_col] if url_col else None,
        "published_time": articles[pub_col] if pub_col else pd.NaT,
    })
    return out[ARTICLE_COLUMNS]


def _build_history_lookup(history_df):
    """user_id -> (article_id_fixed list, impression_time_fixed list)."""
    id_col = "article_id_fixed"
    time_col = "impression_time_fixed"
    ids_series = history_df[id_col] if id_col in history_df.columns else pd.Series([[]] * len(history_df))
    times_series = history_df[time_col] if time_col in history_df.columns else pd.Series([[]] * len(history_df))
    lookup = {}
    for uid, aids, times in zip(history_df["user_id"], ids_series, times_series):
        lookup[uid] = ([str(a) for a in _to_list(aids)], _to_list(times))
    return lookup


def load_ebnerd_interactions(data_dir):
    """Concatenation of train + validation behaviors.parquet, each row's
    history joined in from that split's history.parquet, mapped to the
    unified interactions schema. `split` is left unset (filled in later by
    pipeline.split on the merged, time-sorted pool)."""
    frames = []
    for source in ("train", "validation"):
        split_dir = os.path.join(data_dir, source)
        beh = load_behaviors(split_dir)
        hist_lookup = _build_history_lookup(load_history(split_dir))

        history_article_ids = beh["user_id"].apply(lambda uid: hist_lookup.get(uid, ([], []))[0])
        inview = beh["article_ids_inview"].apply(lambda x: [str(a) for a in _to_list(x)])
        clicked = beh["article_ids_clicked"].apply(lambda x: {str(a) for a in _to_list(x)})
        labels = [
            [1 if aid in clicked_set else 0 for aid in cand]
            for cand, clicked_set in zip(inview, clicked)
        ]

        frames.append(pd.DataFrame({
            "dataset": "ebnerd",
            "impression_id": "ebnerd_" + source + "_" + beh["impression_id"].astype(str),
            "user_id": beh["user_id"],
            "impression_time": beh["impression_time"],
            "history_article_ids": history_article_ids,
            "candidate_article_ids": inview,
            "labels": labels,
            "split": None,
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[INTERACTION_COLUMNS]


def derive_ebnerd_click_history(data_dir):
    """Built directly from history.parquet's (article_id_fixed,
    impression_time_fixed) pairs, which are EB-NeRD's authoritative
    timestamped click log — unioned across train + validation and
    deduplicated, since a user's validation-split history snapshot is a
    superset of their train-split one."""
    frames = []
    for source in ("train", "validation"):
        hist = load_history(os.path.join(data_dir, source))
        df = pd.DataFrame({
            "user_id": hist["user_id"],
            "article_id_fixed": hist["article_id_fixed"].apply(_to_list),
            "impression_time_fixed": hist["impression_time_fixed"].apply(_to_list),
        })
        df = df.explode(["article_id_fixed", "impression_time_fixed"])
        df = df.dropna(subset=["article_id_fixed"])
        frames.append(df.rename(columns={
            "article_id_fixed": "article_id", "impression_time_fixed": "click_time",
        }))

    ch = pd.concat(frames, ignore_index=True)[["user_id", "article_id", "click_time"]]
    ch["article_id"] = ch["article_id"].astype(str)
    ch["click_time"] = pd.to_datetime(ch["click_time"], errors="coerce")
    ch = ch.drop_duplicates(subset=["user_id", "article_id", "click_time"])
    ch.insert(0, "dataset", "ebnerd")
    return ch
