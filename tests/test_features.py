import numpy as np
import pandas as pd

from evidence.config import LEAKAGE_COLUMNS
from evidence.features import (
    CATEGORICAL_FEATURES,
    ISOLATION_NUMERIC,
    NUMERIC_FEATURES,
    add_sequence_features,
    add_time_features,
    build_features,
)
from tests.conftest import make_events


def test_time_features():
    df = make_events(
        [
            ("u", "p", "2024-01-06 03:15:00", "Logon", "logon"),  # Saturday, night
            ("u", "p", "2024-01-08 10:30:00", "Logon", "logon"),  # Monday, work hours
            ("u", "p", "2024-01-08 19:00:00", "Logoff", "logon"),  # boundary -> off hours
        ]
    )
    out = add_time_features(df)
    assert out["hour"].tolist() == [3, 10, 19]
    assert out["day_of_week"].tolist() == [5, 0, 0]
    assert out["is_weekend"].tolist() == [1, 0, 0]
    assert out["off_hours"].tolist() == [1, 0, 1]
    assert out["minute_of_day"].tolist() == [195, 630, 1140]


def test_sequence_features_preserve_order_and_compute_gaps():
    df = make_events(
        [
            ("b", "p2", "2024-01-01 10:00:00", "x", "http"),
            ("a", "p1", "2024-01-01 10:10:00", "x", "http"),
            ("a", "p1", "2024-01-01 10:00:00", "x", "http"),  # out of order on purpose
            ("a", "p9", "2024-01-02 09:00:00", "x", "http"),
        ]
    )
    out = add_sequence_features(df)
    assert out["user"].tolist() == ["b", "a", "a", "a"]  # original order kept
    assert np.isnan(out.loc[0, "gap_prev_sec"])  # first event of user b
    assert np.isnan(out.loc[2, "gap_prev_sec"])  # earliest event of user a
    assert out.loc[1, "gap_prev_sec"] == 600
    assert out.loc[2, "gap_next_sec"] == 600
    assert np.isnan(out.loc[3, "gap_next_sec"])
    assert out["user_day_events"].tolist() == [1, 2, 2, 1]
    assert out["foreign_pc"].tolist() == [0, 0, 0, 1]


def test_no_leakage_columns_in_feature_lists():
    used = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ISOLATION_NUMERIC)
    assert not used & set(LEAKAGE_COLUMNS)


def test_build_features_on_full_data(full_normalized, full_features):
    assert len(full_features) == len(full_normalized)
    for col in NUMERIC_FEATURES + ISOLATION_NUMERIC:
        assert col in full_features.columns
    assert (full_features["gap_prev_sec"].dropna() >= 0).all()
    # Exactly one missing previous-gap per user (their very first event).
    assert full_features["gap_prev_sec"].isna().sum() == full_features["user"].nunique()
    assert full_features["user_day_events"].min() >= 1
    assert set(full_features["foreign_pc"].unique()) <= {0, 1}


def test_build_features_does_not_mutate_input(sample_df):
    before = sample_df.copy()
    build_features(sample_df)
    pd.testing.assert_frame_equal(before, sample_df)
