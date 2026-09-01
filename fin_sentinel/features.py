"""Feature engineering for FinSentinel.

Turns the raw transaction table into the model feature vector defined in
``fin_sentinel.config.MODEL_FEATURES``. Everything here is point-in-time
correct: per-customer baselines only use information from *earlier*
transactions, never from the transaction being scored or from the future.

Usage::

    from fin_sentinel.data_generator import generate_transactions
    from fin_sentinel.features import engineer_features

    features = engineer_features(generate_transactions())
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORY_RISK, MODEL_FEATURES

#: Value used when a customer has no prior transaction history (equivalent to
#: "one week of silence"). Trees and linear models handle this fine.
NO_HISTORY_MINUTES = 7 * 24 * 60.0


def engineer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return ``transactions`` plus the ``MODEL_FEATURES`` columns.

    The input must contain ``customer_id``, ``timestamp``, ``amount``,
    ``category``, ``hour`` and ``day_of_week``.
    """
    df = transactions.sort_values(["customer_id", "timestamp"]).copy()

    # -- per-customer baseline: mean of *prior* transactions only -----------
    grouped = df.groupby("customer_id", sort=False)
    prior_sum = grouped["amount"].cumsum() - df["amount"]
    prior_count = grouped.cumcount()
    df["user_mean_prior"] = np.where(
        prior_count > 0, prior_sum / prior_count.clip(lower=1), df["amount"]
    )
    df["amount_vs_user_mean"] = df["amount"] / df["user_mean_prior"]

    # -- recency -------------------------------------------------------------
    df["minutes_since_prior_txn"] = (
        grouped["timestamp"].diff().dt.total_seconds().div(60).fillna(NO_HISTORY_MINUTES)
    )

    # -- velocity: prior transaction counts in rolling time windows ----------
    # Exact per-customer window counts via searchsorted (no merges, no
    # duplicate-timestamp pitfalls): for each transaction i, count prior
    # transactions j with timestamp in [ts_i - window, ts_i).
    timestamps = df["timestamp"].to_numpy()
    for name, window in (
        ("txn_count_1h", np.timedelta64(1, "h")),
        ("txn_count_24h", np.timedelta64(24, "h")),
    ):
        counts = np.zeros(len(df))
        for _, idx in df.groupby("customer_id", sort=False).indices.items():
            ts = timestamps[idx]
            start = np.searchsorted(ts, ts - window, side="left")
            counts[idx] = np.arange(len(idx)) - start
        df[name] = counts

    # -- calendar / cyclic encodings ------------------------------------------
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["category_risk"] = df["category"].map(CATEGORY_RISK).astype(float)

    if df[MODEL_FEATURES].isna().any().any():
        raise RuntimeError("NaNs produced during feature engineering -- this is a bug")

    return df
