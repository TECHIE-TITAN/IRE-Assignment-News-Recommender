#!/usr/bin/env python3
"""One-command rebuild (Q1): raw MIND files -> unified, temporally-split
processed tables -> feature store.

    python build_pipeline.py

Scope (for now): MIND only, using MIND's own already-temporally-disjoint
files directly as the train/val/test split --
    train = MINDsmall_train   (Nov  9-14, 2019, labeled)
    val   = MINDsmall_dev     (Nov 15,    2019, labeled)
    test  = MINDlarge_test    (Nov 16-22, 2019, UNLABELED -- Codabench only)

Outputs (under --out_dir, default "data"):
    processed/mind/articles.parquet         unified article table (train+val)
    processed/mind/interactions.parquet     unified impression-level behaviors, with `split`
    processed/mind/click_history.parquet    unified timestamped click log, with `split`
    feature_store/mind/article_features.parquet
    feature_store/mind/user_features_val.parquet    (train-period clicks only)
    feature_store/mind/user_features_test.parquet   (train+val-period clicks only)
    reports/mind_pipeline_report.json       row counts + leakage-check result
"""

import argparse
import json
import os
import zipfile

from pipeline import mind as mind_loader
from pipeline.feature_store import build_article_features, build_user_features
from pipeline.split import assert_no_future_click_leakage, assert_split_boundary_monotonic


def _ensure_extracted(data_dir, dir_name):
    """Q1.1 "download" step, scaled down to what's actually needed here: the
    raw MIND zips are large (tens of MB to 600+ MB) and downloading them is
    a deliberate, explicit action (see Assignment.md Part 0) -- this does
    not fetch them over the network. It only auto-extracts a zip that's
    already sitting in `data_dir` if the corresponding folder isn't there
    yet, so re-running the pipeline after a fresh `wget`/`hf download` is
    still one command."""
    target = os.path.join(data_dir, dir_name)
    if os.path.isdir(target) and os.listdir(target):
        return target
    zip_path = os.path.join(data_dir, dir_name + ".zip")
    if os.path.isfile(zip_path):
        print(f"  extracting {zip_path} -> {target} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        return target
    raise FileNotFoundError(
        f"Neither {target}/ nor {zip_path} found. Download the raw MIND files "
        f"first (see Assignment.md Part 0) into {data_dir}/."
    )


def _write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Rebuild the unified MIND data pipeline + feature store from raw files.")
    ap.add_argument("--data_dir", default="data", help="Root dir containing raw MINDsmall_*/MINDlarge_test folders/zips")
    ap.add_argument("--out_dir", default="data", help="Root dir to write processed/, feature_store/, reports/ into")
    args = ap.parse_args()

    # train_dir = _ensure_extracted(args.data_dir, "MINDsmall_train")
    # val_dir = _ensure_extracted(args.data_dir, "MINDsmall_dev")
    train_dir = _ensure_extracted(args.data_dir, "MINDlarge_train")
    val_dir = _ensure_extracted(args.data_dir, "MINDlarge_dev")
    test_dir = _ensure_extracted(args.data_dir, "MINDlarge_test")

    print(f"Loading MIND articles from {train_dir}, {val_dir} ...")
    # Q1 feature store scope is train+val (test is unlabeled and handled
    # separately at prediction time -- see retrieval/ + scripts/generate_predictions.py).
    articles = mind_loader.load_mind_articles(train_dir, val_dir)

    print(f"Loading MIND behaviors from train/val/test ...")
    interactions = mind_loader.load_mind_interactions({"train": train_dir, "val": val_dir, "test": test_dir})
    assert_split_boundary_monotonic(interactions)

    click_history = mind_loader.derive_mind_click_history(interactions)
    click_history = click_history.dropna(subset=["click_time"])

    print(f"\n=== mind: {len(articles)} articles, {len(interactions)} interactions, "
          f"{len(click_history)} click-history rows ===")

    # -- processed (unified, split-tagged) tables ----------------------------
    _write_parquet(articles, os.path.join(args.out_dir, "processed", "mind", "articles.parquet"))
    _write_parquet(interactions, os.path.join(args.out_dir, "processed", "mind", "interactions.parquet"))
    _write_parquet(click_history, os.path.join(args.out_dir, "processed", "mind", "click_history.parquet"))

    # -- feature store ---------------------------------------------------------
    train_interactions = interactions[interactions["split"] == "train"]
    article_features = build_article_features(articles, train_interactions)
    _write_parquet(article_features, os.path.join(args.out_dir, "feature_store", "mind", "article_features.parquet"))

    val_cutoff = interactions.loc[interactions["split"] == "val", "impression_time"].min()
    test_cutoff = interactions.loc[interactions["split"] == "test", "impression_time"].min()
    ch_before_val = click_history[click_history["click_time"] < val_cutoff]
    ch_before_test = click_history[click_history["click_time"] < test_cutoff]
    user_features_val = build_user_features(ch_before_val, val_cutoff, label="mind/user_features_val")
    user_features_test = build_user_features(ch_before_test, test_cutoff, label="mind/user_features_test")
    _write_parquet(user_features_val, os.path.join(args.out_dir, "feature_store", "mind", "user_features_val.parquet"))
    _write_parquet(user_features_test, os.path.join(args.out_dir, "feature_store", "mind", "user_features_test.parquet"))

    # -- leakage guard (Q9) ----------------------------------------------------
    assert_no_future_click_leakage(ch_before_val, val_cutoff, label="mind/user_features_val")
    assert_no_future_click_leakage(ch_before_test, test_cutoff, label="mind/user_features_test")

    # -- report ---------------------------------------------------------------
    split_counts = interactions["split"].value_counts().to_dict()
    report = {
        "dataset": "mind",
        "n_articles": int(len(articles)),
        "n_interactions": int(len(interactions)),
        "n_click_history_rows": int(len(click_history)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "split_boundaries": {
            "train_max_time": str(interactions.loc[interactions["split"] == "train", "impression_time"].max()),
            "val_min_time": str(interactions.loc[interactions["split"] == "val", "impression_time"].min()),
            "val_max_time": str(interactions.loc[interactions["split"] == "val", "impression_time"].max()),
            "test_min_time": str(interactions.loc[interactions["split"] == "test", "impression_time"].min()),
        },
        "n_users_with_val_features": int(len(user_features_val)),
        "n_users_with_test_features": int(len(user_features_test)),
        "leakage_check": "passed",
    }
    report_path = os.path.join(args.out_dir, "reports", "mind_pipeline_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  wrote report -> {report_path}")

    print(f"\nDone. Processed tables -> {args.out_dir}/processed/mind/, "
          f"feature store -> {args.out_dir}/feature_store/mind/, report -> {report_path}")


if __name__ == "__main__":
    main()
