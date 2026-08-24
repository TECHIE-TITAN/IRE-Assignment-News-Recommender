"""Split-boundary assertions (Q9 anti-gaming guards).

MIND's own train/dev/large_test files are already temporally disjoint by
construction (train: Nov 9-14 2019, dev: Nov 15 2019, large_test: Nov 16-22
2019), so pipeline/mind.py assigns `split` directly from which file a row
came from rather than cutting an arbitrary day boundary. What's checked here
is that this assumption actually holds for the files on disk, and that no
future click leaks into a split's feature computation.
"""

import pandas as pd


def assert_split_boundary_monotonic(interactions, time_col="impression_time"):
    """Every train impression must strictly precede every val impression.
    (Test is unlabeled and used only for prediction generation, not offline
    evaluation, but we still check val -> test ordering for completeness.)"""
    for a, b in [("train", "val"), ("val", "test")]:
        max_a = interactions.loc[interactions["split"] == a, time_col].max()
        min_b = interactions.loc[interactions["split"] == b, time_col].min()
        if pd.notna(max_a) and pd.notna(min_b) and max_a > min_b:
            raise AssertionError(
                f"temporal split boundary violated: split '{a}' max impression_time "
                f"({max_a}) is after split '{b}' min impression_time ({min_b})"
            )


def assert_no_future_click_leakage(click_history, cutoff_time, label=""):
    """User/feature-store code must only use clicks strictly before
    `cutoff_time` when computing features for impressions at/after it. Call
    this on whatever click_history slice was actually used to build a
    split's user features."""
    bad = click_history[click_history["click_time"] >= cutoff_time]
    if len(bad):
        raise AssertionError(
            f"{label}: {len(bad)} click-history rows at/after cutoff {cutoff_time} "
            f"leaked into user features (future-click leakage)"
        )
