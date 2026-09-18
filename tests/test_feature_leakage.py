"""A2 Q1/Q9: behaviour-window boundary enforcement for the NEW candidate-level
features in pipeline/features_{common,mind,ebnerd}.py.

This is separate from tests/test_leakage.py (which covers Assignment 1's
click-history/split-boundary guards) because A2 introduces new leakage
surfaces those guards don't touch: EB-NeRD's session-based click count and
past-engagement averages, and both datasets' freshness proxies. Each of
these could leak in a way that would still look like a "reasonable" number
if not caught -- exactly the class of bug that needs a test, not just a
crash, per the lesson in tests/test_leakage.py and design_note.md.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.features_common import (
    compute_first_seen_times,
    days_since_first_seen,
    history_embedding_similarity_map,
    recent_history_with_weights,
    weighted_category_match_map,
)
from pipeline.features_ebnerd import _session_engagement_from_frame


# --------------------------------------------------------------------------
# recent_history_with_weights / weighted_category_match_map: only ever look
# at the history list actually passed in (which is each dataset's own
# already-leakage-safe as-of-this-impression field) -- these tests confirm
# the recent_n truncation and cold-start handling are correct, not that the
# input itself is safe (that's the dataset loaders' job, covered elsewhere).
# --------------------------------------------------------------------------

def test_recent_history_truncates_to_recent_n_and_keeps_order():
    history = ["a", "b", "c", "d", "e"]
    recent, weights = recent_history_with_weights(history, recent_n=3, decay=0.5)
    assert recent == ["c", "d", "e"]
    assert weights[-1] == 1.0  # most recent gets full weight
    assert weights[0] < weights[-1]  # older gets discounted


def test_recent_history_handles_cold_start():
    recent, weights = recent_history_with_weights([], recent_n=20, decay=0.85)
    assert recent == [] and weights == []


def test_weighted_category_match_uses_only_given_history():
    history = ["a", "b", "c"]
    category_lookup = {"a": "sports", "b": "news", "c": "news", "z": "sports"}
    dist = weighted_category_match_map(history, category_lookup, recent_n=20, decay=1.0)
    # "z" (sports) is never in `history`, so it must not influence the
    # distribution even though it shares a category with "a".
    assert set(dist.keys()) <= {"sports", "news"}
    assert dist["news"] > dist["sports"]  # 2 of 3 equally-weighted history articles are "news"


def test_history_embedding_similarity_degrades_to_zero_on_cold_start():
    class _StubIndex:
        def get_embedding(self, aid):
            return None

        def score_candidates(self, user_vec, candidate_ids):
            assert user_vec is None  # cold-start must reach here as None, not a fabricated vector
            return [0.0] * len(candidate_ids)

    scores = history_embedding_similarity_map([], _StubIndex(), ["c1", "c2"])
    assert scores == {"c1": 0.0, "c2": 0.0}


# --------------------------------------------------------------------------
# Freshness proxy (pipeline.features_common.compute_first_seen_times +
# days_since_first_seen): must never let an article's LATER first-appearance
# leak a real, "fresh-looking" day count into an EARLIER impression -- this
# is the actual leakage guard (a per-row point-in-time gate), not a coarse
# split-level restriction on the input (an earlier draft of this module
# tried that; it was correctly safe for val/test looking at train, but NOT
# safe for a split's own internal chronology -- see
# pipeline/features_common.py's docstring for the full reasoning).
# --------------------------------------------------------------------------

def _mind_interactions(rows):
    """rows: list of (impression_time, split, candidate_article_ids)."""
    df = pd.DataFrame(rows, columns=["impression_time", "split", "candidate_article_ids"])
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    return df


def test_first_seen_is_per_article_earliest_appearance_anywhere():
    interactions = _mind_interactions([
        ("2019-11-10 00:00:00", "train", ["N1"]),
        ("2019-11-15 12:00:00", "val", ["N1", "N2"]),
        ("2019-11-17 00:00:00", "test", ["N1", "N2", "N3"]),
    ])
    first_seen = compute_first_seen_times(interactions)
    assert first_seen["N1"] == pd.Timestamp("2019-11-10 00:00:00")
    assert first_seen["N2"] == pd.Timestamp("2019-11-15 12:00:00")
    assert first_seen["N3"] == pd.Timestamp("2019-11-17 00:00:00")


def test_days_since_first_seen_never_credits_a_future_first_appearance():
    """The actual leakage guard: an impression that happens BEFORE an
    article's own (globally-computed) first-seen time must fall back to the
    neutral max-age value, never a real (small, "looks fresh") day count --
    this is what makes it safe that compute_first_seen_times itself is
    computed over the whole table with no split-level restriction."""
    first_seen = pd.Series({
        "N1": pd.Timestamp("2019-11-10 00:00:00"),
        "N2": pd.Timestamp("2019-11-17 00:00:00"),  # N2's true first appearance is LATE
    })
    article_ids = pd.Series(["N1", "N2"])
    # Both rows scored as of an EARLY impression time, strictly before N2's
    # own first-ever appearance.
    impression_times = pd.Series([pd.Timestamp("2019-11-11 00:00:00")] * 2)

    days = days_since_first_seen(article_ids, impression_times, first_seen)
    assert days.iloc[0] == pytest.approx(1.0)  # N1: 1 day after its own first-seen -- a real, legitimate value
    # N2: this impression is BEFORE N2's first-ever appearance in the whole
    # dataset -- must NOT get a negative or "fresh" value; must fall back to
    # the neutral max-observed-age (which here is N1's own 1.0-day value).
    assert days.iloc[1] == days.iloc[0]
    assert days.iloc[1] >= 0


def test_days_since_first_seen_accepts_a_plain_dict_with_missing_keys():
    """pipeline.features_ebnerd.build_ebnerd_candidate_features passes a
    plain dict (a merged published_time / first_seen_time lookup), not a
    Series, and NOT every article_id is guaranteed to be a key in it. A
    dict lookup with gaps is more prone to silently producing an
    `object`-dtype intermediate (Timestamps mixed with plain float NaN)
    that breaks the `.dt` accessor than a Series lookup is -- this
    reproduces that exact shape directly, rather than trusting it by
    inspection alone."""
    first_seen = {"N1": pd.Timestamp("2019-11-10 00:00:00")}  # "N2" deliberately absent
    article_ids = pd.Series(["N1", "N2"])
    impression_times = pd.Series([pd.Timestamp("2019-11-12 00:00:00")] * 2)

    days = days_since_first_seen(article_ids, impression_times, first_seen)
    assert days.iloc[0] == pytest.approx(2.0)  # N1: real, legitimate value
    assert days.iloc[1] == days.iloc[0]  # N2: never in the dict -- neutral fallback, not a crash


# --------------------------------------------------------------------------
# EB-NeRD session/engagement history: the specific anti-gaming trap
# documented in pipeline/features_ebnerd.py -- read_time/scroll_percentage
# and session click counts on a row describe the OUTCOME of that row's own
# impression, not information available beforehand.
# --------------------------------------------------------------------------

def _ebnerd_behaviors(rows):
    """rows: list of (impression_id, user_id, session_id, impression_time,
    read_time, scroll_percentage, n_clicked)."""
    df = pd.DataFrame(rows, columns=[
        "impression_id", "user_id", "session_id", "impression_time",
        "read_time", "scroll_percentage", "n_clicked",
    ])
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    return df


def test_session_click_count_excludes_current_row_and_future_clicks():
    beh = _ebnerd_behaviors([
        ("imp1", "U1", "S1", "2023-05-01 10:00:00", 30.0, 80.0, 1),  # clicks once
        ("imp2", "U1", "S1", "2023-05-01 10:05:00", 20.0, 70.0, 0),  # same session, later
        ("imp3", "U1", "S1", "2023-05-01 10:10:00", 10.0, 60.0, 2),  # same session, later still, clicks twice
    ])
    out = _session_engagement_from_frame(beh)

    # imp1 is the first in its session -- no prior clicks possible.
    assert out.loc["imp1", "session_prior_click_count"] == 0
    # imp2 must see imp1's click (1), but NOT its own n_clicked=0 (moot) or
    # imp3's future clicks.
    assert out.loc["imp2", "session_prior_click_count"] == 1
    # imp3 must see imp1+imp2's clicks (1+0=1), NOT its own n_clicked=2.
    assert out.loc["imp3", "session_prior_click_count"] == 1


def test_user_engagement_average_excludes_current_row_own_value():
    beh = _ebnerd_behaviors([
        ("imp1", "U1", "S1", "2023-05-01 10:00:00", 999.0, 100.0, 0),  # distinctive value
        ("imp2", "U1", "S1", "2023-05-01 10:05:00", 10.0, 10.0, 0),
    ])
    out = _session_engagement_from_frame(beh)

    # imp1 has no PRIOR row at all -- its own read_time=999 must not leak
    # into its own feature (the direct "use the outcome to predict the
    # outcome" trap).
    assert pd.isna(out.loc["imp1", "user_avg_past_read_time"])
    # imp2's average-past-read-time must come from imp1's read_time (999),
    # not from imp2's own read_time (10).
    assert out.loc["imp2", "user_avg_past_read_time"] == 999.0


def test_user_engagement_average_is_per_user_not_global():
    beh = _ebnerd_behaviors([
        ("imp1", "U1", "S1", "2023-05-01 10:00:00", 500.0, 50.0, 0),
        ("imp2", "U2", "S2", "2023-05-01 10:01:00", 100.0, 20.0, 0),  # different user, interleaved in time
        ("imp3", "U1", "S1", "2023-05-01 10:05:00", 10.0, 10.0, 0),
    ])
    out = _session_engagement_from_frame(beh)
    # imp3 (U1) must only average over U1's own prior rows (imp1=500), not
    # U2's imp2 which happened in between chronologically.
    assert out.loc["imp3", "user_avg_past_read_time"] == 500.0
