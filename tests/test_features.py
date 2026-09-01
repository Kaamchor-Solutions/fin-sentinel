"""Tests for the feature engineering pipeline."""

import numpy as np

from fin_sentinel.config import MODEL_FEATURES
from fin_sentinel.data_generator import generate_transactions
from fin_sentinel.features import engineer_features


def _small_features():
    return engineer_features(generate_transactions(n_customers=25, days=12, seed=5))


def test_all_model_features_present_and_clean():
    df = _small_features()
    for col in MODEL_FEATURES:
        assert col in df.columns
    assert not df[MODEL_FEATURES].isna().any().any()
    assert np.isfinite(df[MODEL_FEATURES].to_numpy()).all()


def test_velocity_features_are_non_negative():
    df = _small_features()
    assert (df["txn_count_1h"] >= 0).all()
    assert (df["txn_count_24h"] >= df["txn_count_1h"]).all()


def test_amount_vs_user_mean_is_positive():
    df = _small_features()
    assert (df["amount_vs_user_mean"] > 0).all()


def test_night_flag_matches_hour():
    df = _small_features()
    assert (df.loc[df["hour"] >= 6, "is_night"] == 0).all()
    assert (df.loc[df["hour"] < 6, "is_night"] == 1).all()


def test_category_risk_mapped():
    df = _small_features()
    assert df["category_risk"].between(0, 1).all()
