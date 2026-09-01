"""Tests for the synthetic transaction generator."""

import pandas as pd

from fin_sentinel.data_generator import COLUMNS, generate_transactions


def test_columns_and_labels():
    df = generate_transactions(n_customers=20, days=10, fraud_rate=0.05, seed=1)
    assert list(df.columns) == COLUMNS
    assert set(df["is_fraud"].unique()) == {0, 1}
    assert df["is_fraud"].mean() > 0


def test_fraud_rate_is_in_expected_band():
    df = generate_transactions(n_customers=60, days=20, fraud_rate=0.03, seed=42)
    assert 0.005 < df["is_fraud"].mean() < 0.10


def test_all_three_fraud_patterns_appear():
    df = generate_transactions(n_customers=80, days=30, fraud_rate=0.03, seed=7)
    patterns = set(df.loc[df["is_fraud"] == 1, "fraud_pattern"])
    assert {"card_testing", "big_ticket", "night_owl"} <= patterns
    assert (df.loc[df["is_fraud"] == 0, "fraud_pattern"] == "").all()


def test_timestamps_unique_and_in_window():
    df = generate_transactions(n_customers=30, days=15, seed=3)
    assert df["txn_id"].is_unique
    assert df["timestamp"].between(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-31")).all()


def test_generator_is_reproducible():
    a = generate_transactions(n_customers=10, days=5, seed=99)
    b = generate_transactions(n_customers=10, days=5, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_invalid_arguments_raise():
    import pytest

    with pytest.raises(ValueError):
        generate_transactions(n_customers=1)
    with pytest.raises(ValueError):
        generate_transactions(fraud_rate=1.5)
