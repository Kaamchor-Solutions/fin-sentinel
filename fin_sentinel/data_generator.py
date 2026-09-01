"""Synthetic payment-card transaction generator with realistic fraud patterns.

The generator simulates the spending behaviour of a population of customers
and injects three common fraud patterns on top of it:

1. ``CARD_TESTING`` -- bursts of micro-transactions inside a tight time
   window, used by fraudsters to check whether a stolen card is still live.
2. ``BIG_TICKET``   -- a single purchase far above the customer's normal
   spend.
3. ``NIGHT_OWL``    -- elevated spend concentrated in the 01:00-05:00 window.

Because every transaction is labelled, the output is a ready-to-train
dataset that requires no real customer data -- which makes the whole
pipeline safe to share, fork and demo publicly.

Usage::

    python -m fin_sentinel.data_generator --out data/transactions.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import CATEGORIES

# Relative frequency with which each fraud pattern is injected.
PATTERN_WEIGHTS = {"card_testing": 0.45, "big_ticket": 0.30, "night_owl": 0.25}

COLUMNS = [
    "txn_id",
    "customer_id",
    "timestamp",
    "amount",
    "category",
    "merchant_id",
    "hour",
    "day_of_week",
    "is_fraud",
    "fraud_pattern",
]


def generate_transactions(
    n_customers: int = 500,
    days: int = 60,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labelled synthetic transaction dataset.

    Parameters
    ----------
    n_customers:
        Number of simulated customers.
    days:
        Length of the simulation window in days.
    fraud_rate:
        Approximate share of fraudulent transactions (fraud comes in bursts,
        so the realised rate can drift slightly from this target).
    seed:
        Random seed -- the generator is fully deterministic for a fixed seed.
    """
    if n_customers < 2:
        raise ValueError("n_customers must be at least 2")
    if not 0.0 < fraud_rate < 1.0:
        raise ValueError("fraud_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01")

    # ---- legitimate behaviour -------------------------------------------
    frames: list[pd.DataFrame] = []
    base_amounts = np.empty(n_customers)
    for i in range(n_customers):
        customer_id = f"CUST-{i:05d}"
        base_amount = float(rng.lognormal(mean=3.6, sigma=0.7))  # median ~36
        base_amounts[i] = base_amount
        txn_rate = float(rng.uniform(0.4, 2.8))  # transactions per day
        peak_hour = int(rng.integers(8, 22))
        preferred = rng.choice(len(CATEGORIES), size=int(rng.integers(2, 5)), replace=False)

        counts = rng.poisson(txn_rate, size=days)
        day_idx = np.repeat(np.arange(days), counts)
        n = int(day_idx.size)
        if n == 0:
            continue

        hour = np.clip(np.round(peak_hour + rng.normal(0.0, 3.5, size=n)), 0, 23).astype(int)
        minute = rng.integers(0, 60, size=n)
        second = rng.integers(0, 60, size=n)
        amount = np.round(rng.lognormal(np.log(base_amount), 0.35, size=n), 2)
        category = [CATEGORIES[j] for j in rng.choice(preferred, size=n)]

        frames.append(
            pd.DataFrame(
                {
                    "customer_id": customer_id,
                    "day": day_idx,
                    "hour": hour,
                    "minute": minute,
                    "second": second,
                    "amount": amount,
                    "category": category,
                    "merchant_id": [f"MERCH-{m:04d}" for m in rng.integers(0, 2000, size=n)],
                    "is_fraud": 0,
                    "fraud_pattern": "",
                }
            )
        )

    legit = pd.concat(frames, ignore_index=True)
    target_fraud = int(round(legit["amount"].size * fraud_rate / (1.0 - fraud_rate)))

    # ---- injected fraud ---------------------------------------------------
    fraud_frames: list[pd.DataFrame] = []
    produced = 0
    while produced < max(target_fraud, 1):
        victim = int(rng.integers(0, n_customers))
        customer_id = f"CUST-{victim:05d}"
        base_amount = base_amounts[victim]
        pattern = rng.choice(
            list(PATTERN_WEIGHTS), p=list(PATTERN_WEIGHTS.values())
        )

        if pattern == "card_testing":
            n = int(rng.integers(4, 13))
            hour = int(rng.choice([0, 1, 2, 3, 4, 5, 23]))
            day = int(rng.integers(0, days))
            offset = np.sort(rng.integers(0, 40 * 60, size=n))  # within 40 min
            rows = {
                "customer_id": customer_id,
                "day": np.full(n, day),
                "hour": np.full(n, hour),
                "minute": ((offset // 60) % 60).astype(int),
                "second": (offset % 60).astype(int),
                "amount": np.round(rng.uniform(1.0, 12.0, size=n), 2),
                "category": ["digital_goods" if j % 2 else "subscriptions" for j in range(n)],
                "merchant_id": [f"MERCH-{m:04d}" for m in rng.integers(0, 2000, size=n)],
                "is_fraud": 1,
                "fraud_pattern": "card_testing",
            }
        elif pattern == "big_ticket":
            n = int(rng.integers(1, 3))
            day = int(rng.integers(0, days))
            rows = {
                "customer_id": customer_id,
                "day": np.full(n, day),
                "hour": rng.integers(0, 24, size=n),
                "minute": rng.integers(0, 60, size=n),
                "second": rng.integers(0, 60, size=n),
                "amount": np.round(base_amount * rng.uniform(8.0, 25.0, size=n), 2),
                "category": ["electronics", "travel", "digital_goods"][int(rng.integers(0, 3))],
                "merchant_id": [f"MERCH-{m:04d}" for m in rng.integers(0, 2000, size=n)],
                "is_fraud": 1,
                "fraud_pattern": "big_ticket",
            }
        else:  # night_owl
            n = int(rng.integers(3, 9))
            day = int(rng.integers(0, days))
            rows = {
                "customer_id": customer_id,
                "day": np.full(n, day),
                "hour": rng.integers(1, 6, size=n),
                "minute": rng.integers(0, 60, size=n),
                "second": rng.integers(0, 60, size=n),
                "amount": np.round(base_amount * rng.uniform(1.5, 6.0, size=n), 2),
                "category": [CATEGORIES[j] for j in rng.integers(0, len(CATEGORIES), size=n)],
                "merchant_id": [f"MERCH-{m:04d}" for m in rng.integers(0, 2000, size=n)],
                "is_fraud": 1,
                "fraud_pattern": "night_owl",
            }

        fraud_frames.append(pd.DataFrame(rows))
        produced += n

    fraud = pd.concat(fraud_frames, ignore_index=True)

    # ---- assemble ---------------------------------------------------------
    df = pd.concat([legit, fraud], ignore_index=True)
    df["timestamp"] = start + pd.to_timedelta(df["day"], unit="D") + pd.to_timedelta(
        df["hour"], unit="h"
    ) + pd.to_timedelta(df["minute"], unit="m") + pd.to_timedelta(df["second"], unit="s")
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = df.drop(columns=["day"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["txn_id"] = [f"TXN-{i:07d}" for i in range(len(df))]
    return df[COLUMNS]


def save_transactions(df: pd.DataFrame, path) -> None:
    """Write the dataset to CSV, creating parent directories as needed."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic transactions")
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/transactions.csv")
    args = parser.parse_args()

    data = generate_transactions(
        n_customers=args.customers, days=args.days, fraud_rate=args.fraud_rate, seed=args.seed
    )
    save_transactions(data, args.out)
    rate = data["is_fraud"].mean()
    print(f"Wrote {len(data):,} transactions ({rate:.2%} fraud) to {args.out}")
