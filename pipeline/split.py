"""Temporal train/val/test splitting and the leakage assertions that guard it.

Splitting is always time-based, never random, per the assignment spec: the
last `test_days` days of a dataset's merged interaction pool become `test`,
the preceding `val_days` days become `val`, and everything earlier is
`train`. Both loaders (pipeline/mind.py, pipeline/ebnerd.py) merge each
dataset's own train+dev/validation bundles into one pool before this cut is
applied, so the split boundary is computed on the dataset's actual full
timeline rather than reusing the raw files' own train/dev split as-is.
"""

import numpy as np
import pandas as pd

from pipeline.schema import SPLITS


def assign_split_by_day(times, val_start_day, test_start_day):
    """Vectorized split assignment from precomputed day boundaries. Shared by
    `temporal_split` (on interactions) and by build_pipeline.py (to tag
    click_history rows with the same boundaries, for diagnostics)."""
    day = pd.to_datetime(times).dt.floor("D")
    return np.where(day >= pd.Timestamp(test_start_day), "test",
                     np.where(day >= pd.Timestamp(val_start_day), "val", "train"))


def temporal_split(interactions, time_col="impression_time", val_days=1, test_days=1):
    """Adds/overwrites a `split` column on a copy of `interactions`. Returns
    (df_with_split, boundaries_dict). `boundaries["val_start_day"]` and
    `["test_start_day"]` are pd.Timestamp — reuse them as the leakage-safe
    cutoff instants for user-feature snapshots and for tagging click_history."""
    day = interactions[time_col].dt.floor("D")
    max_day = day.max()
    test_start_day = max_day - pd.Timedelta(days=test_days - 1)
    val_start_day = test_start_day - pd.Timedelta(days=val_days)

    out = interactions.copy()
    out["split"] = assign_split_by_day(interactions[time_col], val_start_day, test_start_day)
    boundaries = {
        "min_day": day.min(),
        "max_day": max_day,
        "val_start_day": val_start_day,
        "test_start_day": test_start_day,
    }
    return out, boundaries


def validate_split_counts(interactions):
    """Raises if any split ended up empty (bad --val_days/--test_days for
    this dataset's date range). Returns the per-split row counts."""
    counts = interactions["split"].value_counts().to_dict()
    for s in SPLITS:
        if counts.get(s, 0) == 0:
            raise ValueError(
                f"split '{s}' is empty after temporal_split (got {counts}); "
                f"adjust --val_days/--test_days for this dataset's date range"
            )
    return {s: int(counts.get(s, 0)) for s in SPLITS}


def assert_split_boundary_monotonic(interactions, time_col="impression_time"):
    """Q9 guard: every train impression must strictly precede every val
    impression, which must strictly precede every test impression."""
    for a, b in [("train", "val"), ("val", "test")]:
        max_a = interactions.loc[interactions["split"] == a, time_col].max()
        min_b = interactions.loc[interactions["split"] == b, time_col].min()
        if pd.notna(max_a) and pd.notna(min_b) and max_a > min_b:
            raise AssertionError(
                f"temporal split boundary violated: split '{a}' max impression_time "
                f"({max_a}) is after split '{b}' min impression_time ({min_b})"
            )


def assert_no_future_click_leakage(click_history, cutoff_time, label=""):
    """Q9 guard: user/feature-store code must only use clicks strictly before
    `cutoff_time` when computing features for impressions at/after it. Call
    this on whatever click_history slice was actually used to build a split's
    user features."""
    bad = click_history[click_history["click_time"] >= cutoff_time]
    if len(bad):
        raise AssertionError(
            f"{label}: {len(bad)} click-history rows at/after cutoff {cutoff_time} "
            f"leaked into user features (future-click leakage)"
        )
