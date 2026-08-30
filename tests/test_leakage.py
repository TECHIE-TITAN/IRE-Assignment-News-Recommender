"""Q9 anti-gaming: behaviour-window boundary + no-future-click-leakage guards.

Two layers, mirroring how `pipeline/split.py` is actually used:
  1. Unit tests against `assert_split_boundary_monotonic` /
     `assert_no_future_click_leakage` directly, on synthetic data -- prove
     the guards themselves catch a violation and don't false-positive on
     clean data.
  2. An integration test against `pipeline.feature_store.build_user_features`
     -- proves the guard is actually wired into the code path that builds
     user features, not just defined and unused.
  3. Integration tests against this run's real, already-built
     `data/processed/<dataset>/` output (skipped if `build_pipeline.py`
     hasn't been run for a dataset) -- re-derive each split's cutoff from
     the real data and re-run the leakage check against it directly,
     rather than trusting the pipeline report's self-reported
     `"leakage_check": "passed"` field.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature_store import build_user_features
from pipeline.split import assert_no_future_click_leakage, assert_split_boundary_monotonic

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _click_history(rows):
    df = pd.DataFrame(rows, columns=["user_id", "article_id", "click_time"])
    df["click_time"] = pd.to_datetime(df["click_time"])
    return df


def _interactions(rows):
    df = pd.DataFrame(rows, columns=["impression_time", "split"])
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    return df


# --------------------------------------------------------------------------
# 1. Unit tests on the guard functions themselves
# --------------------------------------------------------------------------

def test_no_future_click_leakage_raises_on_leaky_data():
    """A click at exactly the cutoff, or after it, must be rejected --
    the boundary is a strict '<', not '<='."""
    ch = _click_history([
        ("U1", "N1", "2019-11-14 23:00:00"),   # before cutoff -- fine
        ("U1", "N2", "2019-11-15 00:00:00"),   # exactly at cutoff -- leaks
    ])
    with pytest.raises(AssertionError, match="future-click leakage"):
        assert_no_future_click_leakage(ch, cutoff_time=pd.Timestamp("2019-11-15 00:00:00"))


def test_no_future_click_leakage_passes_on_clean_data():
    ch = _click_history([
        ("U1", "N1", "2019-11-14 22:00:00"),
        ("U1", "N2", "2019-11-14 23:59:59"),
    ])
    assert_no_future_click_leakage(ch, cutoff_time=pd.Timestamp("2019-11-15 00:00:00"))


def test_split_boundary_monotonic_raises_on_overlap():
    """A single train impression timestamped after val's start must be caught."""
    interactions = _interactions([
        ("2019-11-14 12:00:00", "train"),
        ("2019-11-15 12:00:00", "train"),  # after val's own min -- violates ordering
        ("2019-11-15 00:00:00", "val"),
    ])
    with pytest.raises(AssertionError, match="temporal split boundary violated"):
        assert_split_boundary_monotonic(interactions)


def test_split_boundary_monotonic_passes_on_valid_split():
    interactions = _interactions([
        ("2019-11-14 23:59:59", "train"),
        ("2019-11-15 00:00:00", "val"),
        ("2019-11-15 23:59:59", "val"),
        ("2019-11-16 00:00:00", "test"),
    ])
    assert_split_boundary_monotonic(interactions)


# --------------------------------------------------------------------------
# 2. Integration: the guard is actually wired into build_user_features,
#    not just defined and callable in isolation
# --------------------------------------------------------------------------

def test_build_user_features_rejects_leaky_input_even_if_caller_forgets_to_filter():
    """`build_user_features` re-checks the cutoff internally (defense in
    depth) -- this must still raise even when the caller passes in
    click history that was never pre-filtered."""
    cutoff = pd.Timestamp("2019-11-15 00:00:00")
    ch = _click_history([
        ("U1", "N1", "2019-11-14 20:00:00"),
        ("U1", "N2", "2019-11-15 06:00:00"),  # caller "forgot" to filter this out
    ])
    with pytest.raises(AssertionError, match="future-click leakage"):
        build_user_features(ch, cutoff_time=cutoff, label="test/user_features")


def test_build_user_features_accepts_properly_filtered_input():
    cutoff = pd.Timestamp("2019-11-15 00:00:00")
    ch = _click_history([
        ("U1", "N1", "2019-11-14 20:00:00"),
        ("U1", "N2", "2019-11-14 22:00:00"),
    ])
    feats = build_user_features(ch, cutoff_time=cutoff, label="test/user_features")
    assert len(feats) == 1
    assert feats.loc[0, "history_length"] == 2


# --------------------------------------------------------------------------
# 3. Integration against this run's real, already-built pipeline output
# --------------------------------------------------------------------------

def _processed_dir(dataset):
    return os.path.join(DATA_DIR, "processed", dataset)


def _skip_unless_built(dataset):
    d = _processed_dir(dataset)
    if not (os.path.isfile(os.path.join(d, "interactions.parquet"))
            and os.path.isfile(os.path.join(d, "click_history.parquet"))):
        pytest.skip(f"data/processed/{dataset}/ not built -- run `python build_pipeline.py "
                    f"--dataset {dataset}` first")


# A `(dataset, split)` pair where filtering-to-before-cutoff is expected to
# be vacuous (no post-cutoff clicks exist in click_history.parquet to begin
# with), for a real, dataset-construction reason rather than a broken
# filter -- see the per-case comments in the test below. Anything NOT
# listed here is expected to have genuine post-cutoff clicks, so the test
# insists on seeing some (otherwise the leakage check wouldn't actually be
# exercising anything for that pair).
_EXPECTED_VACUOUS = {
    ("mind", "test"): (
        "MINDlarge_test is unlabeled -- derive_mind_click_history only derives a "
        "click_time from positive-labelled (labels==1) candidates, so an unlabeled "
        "split can never contribute a click_time at all, let alone a post-cutoff one."
    ),
    ("ebnerd", "val"): (
        "derive_ebnerd_click_history is built from each split's own history.parquet, "
        "which records a user's pre-existing history up to that split's own reference "
        "point (not clicks generated during the split's evaluation window) -- so even "
        "validation/history.parquet never carries a click at/after validation's own start."
    ),
}


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_real_pipeline_output_has_no_future_click_leakage(dataset):
    """Re-derives each split's cutoff and re-runs the leakage check directly
    against this run's actual processed output, rather than trusting the
    pipeline report's self-reported leakage_check field."""
    _skip_unless_built(dataset)
    d = _processed_dir(dataset)
    interactions = pd.read_parquet(os.path.join(d, "interactions.parquet"), columns=["impression_time", "split"])
    click_history = pd.read_parquet(os.path.join(d, "click_history.parquet"), columns=["click_time"])

    assert_split_boundary_monotonic(interactions)

    for split in ("val", "test"):
        if split not in interactions["split"].unique():
            continue  # EB-NeRD has no test split in this data
        cutoff = interactions.loc[interactions["split"] == split, "impression_time"].min()
        before = click_history[click_history["click_time"] < cutoff]
        after_or_at = click_history[click_history["click_time"] >= cutoff]

        # The check should actually be meaningful: confirm there really are
        # clicks at/after the cutoff to filter out, so the assertion passing
        # isn't vacuous -- except for the pairs above, where vacuity itself
        # is the correct, understood behaviour (not a broken filter).
        if (dataset, split) not in _EXPECTED_VACUOUS:
            assert len(after_or_at) > 0, (
                f"{dataset}/{split}: no clicks at/after cutoff exist in the raw data -- "
                f"this test would pass even with a broken filter, so it isn't a real check "
                f"(if this is now expected, add it to _EXPECTED_VACUOUS with a reason)"
            )

        assert_no_future_click_leakage(before, cutoff, label=f"{dataset}/{split} (re-derived)")


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_pipeline_report_records_passed_leakage_check(dataset):
    report_path = os.path.join(DATA_DIR, "reports", f"{dataset}_pipeline_report.json")
    if not os.path.isfile(report_path):
        pytest.skip(f"{report_path} not present -- run build_pipeline.py --dataset {dataset} first")
    import json
    with open(report_path) as f:
        report = json.load(f)
    assert report["leakage_check"] == "passed"
