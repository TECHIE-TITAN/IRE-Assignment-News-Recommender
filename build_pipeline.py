#!/usr/bin/env python3
"""One-command rebuild: raw MIND / EB-NeRD files -> unified, temporally-split
processed tables -> feature store.

    python build_pipeline.py
    python build_pipeline.py --datasets mind
    python build_pipeline.py --datasets ebnerd --ebnerd_bundle ebnerd_demo
    python build_pipeline.py --mind_val_days 1 --mind_test_days 1

Outputs (under --out_dir, default "data"):
    processed/<dataset>/articles.parquet        unified article table
    processed/<dataset>/interactions.parquet    unified impression-level behaviors, with `split`
    processed/<dataset>/click_history.parquet   unified timestamped click log, with `split`
    feature_store/<dataset>/article_features.parquet
    feature_store/<dataset>/user_features_val.parquet    (train-period clicks only)
    feature_store/<dataset>/user_features_test.parquet   (train+val-period clicks only)
    reports/<dataset>_pipeline_report.json      row counts + split boundaries
"""

import argparse
import json
import os

from pipeline import ebnerd as ebnerd_loader
from pipeline import mind as mind_loader
from pipeline.feature_store import build_article_features, build_user_features
from pipeline.split import (
    assert_split_boundary_monotonic,
    assign_split_by_day,
    temporal_split,
    validate_split_counts,
)


def _write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def process_dataset(name, articles, interactions, click_history, out_dir, val_days, test_days):
    print(f"\n=== {name}: {len(articles)} articles, {len(interactions)} interactions, "
          f"{len(click_history)} click-history rows ===")

    interactions, boundaries = temporal_split(interactions, val_days=val_days, test_days=test_days)
    split_counts = validate_split_counts(interactions)
    assert_split_boundary_monotonic(interactions)
    print(f"  split boundaries: {boundaries}")
    print(f"  split counts: {split_counts}")

    click_history = click_history.dropna(subset=["click_time"]).copy()
    click_history["split"] = assign_split_by_day(
        click_history["click_time"], boundaries["val_start_day"], boundaries["test_start_day"])

    # -- processed (unified, split-tagged) tables ----------------------------
    _write_parquet(articles, os.path.join(out_dir, "processed", name, "articles.parquet"))
    _write_parquet(interactions, os.path.join(out_dir, "processed", name, "interactions.parquet"))
    _write_parquet(click_history, os.path.join(out_dir, "processed", name, "click_history.parquet"))

    # -- feature store ---------------------------------------------------------
    train_interactions = interactions[interactions["split"] == "train"]
    article_features = build_article_features(articles, train_interactions)
    _write_parquet(article_features, os.path.join(out_dir, "feature_store", name, "article_features.parquet"))

    val_cutoff = boundaries["val_start_day"]
    test_cutoff = boundaries["test_start_day"]
    ch_before_val = click_history[click_history["click_time"] < val_cutoff]
    ch_before_test = click_history[click_history["click_time"] < test_cutoff]
    user_features_val = build_user_features(ch_before_val, val_cutoff, label=f"{name}/user_features_val")
    user_features_test = build_user_features(ch_before_test, test_cutoff, label=f"{name}/user_features_test")
    _write_parquet(user_features_val, os.path.join(out_dir, "feature_store", name, "user_features_val.parquet"))
    _write_parquet(user_features_test, os.path.join(out_dir, "feature_store", name, "user_features_test.parquet"))

    # -- report ---------------------------------------------------------------
    report = {
        "dataset": name,
        "n_articles": int(len(articles)),
        "n_interactions": int(len(interactions)),
        "n_click_history_rows": int(len(click_history)),
        "split_counts": split_counts,
        "split_boundaries": boundaries,
        "n_users_with_val_features": int(len(user_features_val)),
        "n_users_with_test_features": int(len(user_features_test)),
    }
    report_path = os.path.join(out_dir, "reports", f"{name}_pipeline_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  wrote report -> {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Rebuild the unified data pipeline + feature store from raw files.")
    ap.add_argument("--data_dir", default="data", help="Root dir containing raw MIND*/ebnerd_* folders")
    ap.add_argument("--out_dir", default="data", help="Root dir to write processed/, feature_store/, reports/ into")
    ap.add_argument("--datasets", default="mind,ebnerd", help="Comma-separated subset: mind,ebnerd")
    ap.add_argument("--ebnerd_bundle", default="ebnerd_small", choices=["ebnerd_demo", "ebnerd_small"],
                     help="Which EB-NeRD bundle to build from (ebnerd_demo is much faster for a first test run)")
    ap.add_argument("--mind_val_days", type=int, default=1)
    ap.add_argument("--mind_test_days", type=int, default=1)
    ap.add_argument("--ebnerd_val_days", type=int, default=1)
    ap.add_argument("--ebnerd_test_days", type=int, default=1)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if "mind" in datasets:
        train_dir = os.path.join(args.data_dir, "MINDsmall_train")
        dev_dir = os.path.join(args.data_dir, "MINDsmall_dev")
        print(f"Loading MIND from {train_dir}, {dev_dir} ...")
        articles = mind_loader.load_mind_articles(train_dir, dev_dir)
        interactions = mind_loader.load_mind_interactions(train_dir, dev_dir)
        click_history = mind_loader.derive_mind_click_history(interactions)
        process_dataset("mind", articles, interactions, click_history,
                         args.out_dir, args.mind_val_days, args.mind_test_days)

    if "ebnerd" in datasets:
        ebnerd_dir = os.path.join(args.data_dir, args.ebnerd_bundle)
        print(f"Loading EB-NeRD from {ebnerd_dir} ...")
        articles = ebnerd_loader.load_ebnerd_articles(ebnerd_dir)
        interactions = ebnerd_loader.load_ebnerd_interactions(ebnerd_dir)
        click_history = ebnerd_loader.derive_ebnerd_click_history(ebnerd_dir)
        process_dataset("ebnerd", articles, interactions, click_history,
                         args.out_dir, args.ebnerd_val_days, args.ebnerd_test_days)

    print(f"\nDone. Processed tables -> {args.out_dir}/processed/<dataset>/, "
          f"feature store -> {args.out_dir}/feature_store/<dataset>/, reports -> {args.out_dir}/reports/")


if __name__ == "__main__":
    main()
