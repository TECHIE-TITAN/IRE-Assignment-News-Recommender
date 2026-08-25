#!/usr/bin/env python3
"""One-command rebuild (Q1): raw MIND or EB-NeRD files -> unified,
temporally-split processed tables -> feature store.

    python build_pipeline.py --dataset mind
    python build_pipeline.py --dataset ebnerd

Scope (for now):
  MIND: MINDsmall_train (train) / MINDsmall_dev (val) / MINDlarge_test
  (test, UNLABELED -- Codabench only) -- MIND's own files, already
  temporally disjoint by construction, used directly as the split.

  EB-NeRD: ebnerd_small's train/ (train) and validation/ (val) --
  likewise already temporally disjoint by construction (train ends
  2023-05-25 07:00, validation starts the same instant). The large,
  unlabeled ebnerd_testset (13.5M impressions) is NOT processed here --
  unlike MIND's build_pipeline.py, which does fold MINDlarge_test into
  this same interactions.parquet (harmless at 2.37M rows, though nothing
  downstream actually reads that portion of it back out -- Q5 streams the
  raw test file directly either way). EB-NeRD's test set is ~5.7x larger
  and needs a per-impression history join that's expensive at that scale,
  so scripts/generate_predictions_ebnerd.py handles it entirely
  separately, streamed from raw parquet.

Outputs (under --out_dir, default "data"):
    processed/<dataset>/articles.parquet         unified article table (train+val)
    processed/<dataset>/interactions.parquet     unified impression-level behaviors, with `split`
    processed/<dataset>/click_history.parquet    unified timestamped click log, with `split`
    feature_store/<dataset>/article_features.parquet
    feature_store/<dataset>/user_features_val.parquet    (train-period clicks only)
    feature_store/<dataset>/user_features_test.parquet   (train+val-period clicks only; MIND only, see above)
    reports/<dataset>_pipeline_report.json       row counts + leakage-check result
"""

import argparse
import json
import os
import zipfile

from pipeline import ebnerd as ebnerd_loader
from pipeline import mind as mind_loader
from pipeline.feature_store import build_article_features, build_user_features
from pipeline.split import assert_no_future_click_leakage, assert_split_boundary_monotonic


def _ensure_extracted(data_dir, dir_name):
    """Q1.1 "download" step, scaled down to what's actually needed here: the
    raw zips are large (tens of MB to 1.6+ GB) and downloading them is a
    deliberate, explicit action (see Assignment.md Part 0) -- this does not
    fetch them over the network. It only auto-extracts a zip that's already
    sitting in `data_dir` if the corresponding folder isn't there yet, so
    re-running the pipeline after a fresh download is still one command."""
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
        f"Neither {target}/ nor {zip_path} found. Download the raw files first "
        f"(see Assignment.md Part 0) into {data_dir}/."
    )


def _write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def build_mind(args):
    train_dir = _ensure_extracted(args.data_dir, args.mind_train)
    val_dir = _ensure_extracted(args.data_dir, args.mind_val)
    test_dir = _ensure_extracted(args.data_dir, args.mind_test)

    print(f"Loading MIND articles from {train_dir}, {val_dir} ...")
    articles = mind_loader.load_mind_articles(train_dir, val_dir)

    print("Loading MIND behaviors from train/val/test ...")
    interactions = mind_loader.load_mind_interactions({"train": train_dir, "val": val_dir, "test": test_dir})
    assert_split_boundary_monotonic(interactions)

    click_history = mind_loader.derive_mind_click_history(interactions)
    click_history = click_history.dropna(subset=["click_time"])
    return articles, interactions, click_history, {"has_test_split": True}


def build_ebnerd(args):
    ebnerd_dir = _ensure_extracted(args.data_dir, args.ebnerd_bundle)

    print(f"Loading EB-NeRD articles from {ebnerd_dir} ...")
    articles = ebnerd_loader.load_ebnerd_articles(ebnerd_dir)

    print("Loading EB-NeRD behaviors from train/validation ...")
    interactions = ebnerd_loader.load_ebnerd_interactions(ebnerd_dir)
    assert_split_boundary_monotonic(interactions)

    click_history = ebnerd_loader.derive_ebnerd_click_history(ebnerd_dir)
    click_history = click_history.dropna(subset=["click_time"])
    return articles, interactions, click_history, {"has_test_split": False}


def main():
    ap = argparse.ArgumentParser(description="Rebuild the unified data pipeline + feature store from raw files.")
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    ap.add_argument("--data_dir", default="data", help="Root dir containing raw dataset folders/zips")
    ap.add_argument("--out_dir", default="data", help="Root dir to write processed/, feature_store/, reports/ into")
    ap.add_argument("--mind_train", default="MINDlarge_train", help="MIND only; e.g. MINDsmall_train for the small bundle")
    ap.add_argument("--mind_val", default="MINDlarge_dev", help="MIND only; e.g. MINDsmall_dev for the small bundle")
    ap.add_argument("--mind_test", default="MINDlarge_test", help="MIND only -- unlabeled, Codabench only")
    ap.add_argument("--ebnerd_bundle", default="ebnerd_small", help="EB-NeRD only; the demo/small bundle dir name")
    args = ap.parse_args()

    if args.dataset == "mind":
        articles, interactions, click_history, meta = build_mind(args)
    else:
        articles, interactions, click_history, meta = build_ebnerd(args)

    print(f"\n=== {args.dataset}: {len(articles)} articles, {len(interactions)} interactions, "
          f"{len(click_history)} click-history rows ===")

    # -- processed (unified, split-tagged) tables ----------------------------
    _write_parquet(articles, os.path.join(args.out_dir, "processed", args.dataset, "articles.parquet"))
    _write_parquet(interactions, os.path.join(args.out_dir, "processed", args.dataset, "interactions.parquet"))
    _write_parquet(click_history, os.path.join(args.out_dir, "processed", args.dataset, "click_history.parquet"))

    # -- feature store ---------------------------------------------------------
    train_interactions = interactions[interactions["split"] == "train"]
    article_features = build_article_features(articles, train_interactions)
    _write_parquet(article_features, os.path.join(args.out_dir, "feature_store", args.dataset, "article_features.parquet"))

    val_cutoff = interactions.loc[interactions["split"] == "val", "impression_time"].min()
    ch_before_val = click_history[click_history["click_time"] < val_cutoff]
    user_features_val = build_user_features(ch_before_val, val_cutoff, label=f"{args.dataset}/user_features_val")
    _write_parquet(user_features_val, os.path.join(args.out_dir, "feature_store", args.dataset, "user_features_val.parquet"))
    assert_no_future_click_leakage(ch_before_val, val_cutoff, label=f"{args.dataset}/user_features_val")

    n_users_with_test_features = None
    if meta["has_test_split"]:
        test_cutoff = interactions.loc[interactions["split"] == "test", "impression_time"].min()
        ch_before_test = click_history[click_history["click_time"] < test_cutoff]
        user_features_test = build_user_features(ch_before_test, test_cutoff, label=f"{args.dataset}/user_features_test")
        _write_parquet(user_features_test, os.path.join(args.out_dir, "feature_store", args.dataset, "user_features_test.parquet"))
        assert_no_future_click_leakage(ch_before_test, test_cutoff, label=f"{args.dataset}/user_features_test")
        n_users_with_test_features = int(len(user_features_test))

    # -- report ---------------------------------------------------------------
    split_counts = interactions["split"].value_counts().to_dict()
    boundaries = {
        "train_max_time": str(interactions.loc[interactions["split"] == "train", "impression_time"].max()),
        "val_min_time": str(interactions.loc[interactions["split"] == "val", "impression_time"].min()),
        "val_max_time": str(interactions.loc[interactions["split"] == "val", "impression_time"].max()),
    }
    if meta["has_test_split"]:
        boundaries["test_min_time"] = str(interactions.loc[interactions["split"] == "test", "impression_time"].min())
    report = {
        "dataset": args.dataset,
        "n_articles": int(len(articles)),
        "n_interactions": int(len(interactions)),
        "n_click_history_rows": int(len(click_history)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "split_boundaries": boundaries,
        "n_users_with_val_features": int(len(user_features_val)),
        "n_users_with_test_features": n_users_with_test_features,
        "leakage_check": "passed",
    }
    report_path = os.path.join(args.out_dir, "reports", f"{args.dataset}_pipeline_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  wrote report -> {report_path}")

    print(f"\nDone. Processed tables -> {args.out_dir}/processed/{args.dataset}/, "
          f"feature store -> {args.out_dir}/feature_store/{args.dataset}/, report -> {report_path}")


if __name__ == "__main__":
    main()
