"""Builds the article-feature and user-feature tables that make up the
"feature store" deliverable.

Leakage discipline:
  - Article popularity stats are computed strictly from the `train` split's
    interactions, never val/test, so article features don't encode
    information from the future relative to when they'd be used at serving
    time.
  - User features are snapshotted per split boundary: `user_features_val` is
    built only from clicks strictly before val starts (i.e. train-period
    clicks), and `user_features_test` only from clicks strictly before test
    starts (i.e. train+val-period clicks). `pipeline.split.assert_no_future_click_leakage`
    is re-checked inside `build_user_features` as a defense-in-depth guard on
    top of the caller's own filtering.
"""

import numpy as np

from pipeline.split import assert_no_future_click_leakage


def _word_count(series):
    return series.fillna("").apply(lambda s: len(str(s).split()))


def build_article_features(articles, train_interactions):
    """`embedding` is left as a null placeholder column here — Q3 populates
    it later by computing/loading article embeddings; Q1's feature store just
    reserves the slot."""
    feats = articles.copy()
    feats["n_title_words"] = _word_count(feats["title"])
    feats["n_abstract_words"] = _word_count(feats["abstract"])
    feats["n_body_words"] = _word_count(feats["body"])
    feats["n_entities"] = feats["entities"].apply(lambda e: len(e) if isinstance(e, list) else 0)

    exploded = train_interactions[["candidate_article_ids", "labels"]].explode(
        ["candidate_article_ids", "labels"]
    ).dropna(subset=["candidate_article_ids"])
    exploded = exploded.rename(columns={"candidate_article_ids": "article_id"})
    exposures = exploded.groupby("article_id").size()
    clicks = exploded[exploded["labels"] == 1].groupby("article_id").size()

    feats["train_impressions"] = feats["article_id"].map(exposures).fillna(0).astype(int)
    feats["train_clicks"] = feats["article_id"].map(clicks).fillna(0).astype(int)
    feats["train_ctr"] = np.where(
        feats["train_impressions"] > 0,
        feats["train_clicks"] / feats["train_impressions"].replace(0, np.nan),
        np.nan,
    )
    feats["embedding"] = None
    return feats


def build_user_features(click_history_before_cutoff, cutoff_time, label=""):
    """`click_history_before_cutoff` must already contain only clicks
    strictly before `cutoff_time` (caller filters); this re-asserts that
    invariant before computing recency features."""
    assert_no_future_click_leakage(click_history_before_cutoff, cutoff_time, label)

    g = click_history_before_cutoff.groupby("user_id")
    feats = g.agg(
        history_length=("article_id", "count"),
        distinct_articles_clicked=("article_id", "nunique"),
        last_click_time=("click_time", "max"),
        first_click_time=("click_time", "min"),
    ).reset_index()
    feats["recency_hours"] = (cutoff_time - feats["last_click_time"]).dt.total_seconds() / 3600.0
    feats["tenure_days"] = (feats["last_click_time"] - feats["first_click_time"]).dt.total_seconds() / 86400.0
    return feats
