"""Training pipeline for FinSentinel.

Generates (or loads) synthetic data, engineers features, trains three
candidate models, evaluates them on a **time-based** holdout (the last 20%
of the timeline, mimicking a real deployment where the future is unknown),
picks the best by PR-AUC and writes:

* ``artifacts/model.joblib``  -- bundle: model, name, threshold, features
* ``artifacts/metrics.json``  -- full evaluation report
* ``data/transactions.csv``   -- the generated dataset

Usage::

    python -m fin_sentinel.train                      # full run (~1 min)
    python -m fin_sentinel.train --quick              # tiny smoke run
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, MODEL_FEATURES
from .data_generator import generate_transactions, save_transactions
from .features import engineer_features


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the score threshold that maximises F1 on the training data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1]))  # last precision/recall pair has no threshold
    return float(thresholds[best])


def _candidate_models(fraud_rate: float, seed: int) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "isolation_forest": IsolationForest(
            n_estimators=300, contamination=fraud_rate, random_state=seed
        ),
    }


def train_pipeline(
    n_customers: int = 500,
    days: int = 60,
    fraud_rate: float = 0.02,
    seed: int = 42,
    artifact_dir=DEFAULT_ARTIFACT_DIR,
    data_dir=DEFAULT_DATA_DIR,
    quick: bool = False,
) -> dict:
    """Run the full train -> evaluate -> select -> persist pipeline.

    Returns the metrics dictionary that is also written to ``metrics.json``.
    """
    if quick:  # small enough for the test-suite to stay fast
        n_customers, days = 40, 20

    from pathlib import Path

    artifact_dir = Path(artifact_dir)
    data_dir = Path(data_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------
    transactions = generate_transactions(
        n_customers=n_customers, days=days, fraud_rate=fraud_rate, seed=seed
    )
    save_transactions(transactions, data_dir / "transactions.csv")
    features = engineer_features(transactions)

    # ---- time-based split: train on the past, test on the future -----------
    features = features.sort_values("timestamp").reset_index(drop=True)
    split = int(len(features) * 0.8)
    train, test = features.iloc[:split], features.iloc[split:]
    X_train, y_train = train[MODEL_FEATURES], train["is_fraud"].to_numpy()
    X_test, y_test = test[MODEL_FEATURES], test["is_fraud"].to_numpy()

    # ---- train & evaluate ---------------------------------------------------
    results = {}
    fitted = {}
    for name, model in _candidate_models(fraud_rate, seed).items():
        if name == "isolation_forest":
            model.fit(X_train)  # unsupervised: labels not used
            score_train = -model.decision_function(X_train)
            score_test = -model.decision_function(X_test)
        else:
            model.fit(X_train, y_train)
            score_train = model.predict_proba(X_train)[:, 1]
            score_test = model.predict_proba(X_test)[:, 1]

        threshold = (
            float(np.quantile(score_train, 1 - fraud_rate))
            if name == "isolation_forest"
            else _best_f1_threshold(y_train, score_train)
        )
        preds = (score_test >= threshold).astype(int)
        results[name] = {
            "average_precision": round(float(average_precision_score(y_test, score_test)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, score_test)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "threshold": round(threshold, 4),
        }
        fitted[name] = model

    best_name = max(results, key=lambda n: results[n]["average_precision"])

    # ---- permutation importance (supervised winners only) ------------------
    importance: dict[str, float] = {}
    if best_name != "isolation_forest":
        perm = permutation_importance(
            fitted[best_name], X_test, y_test, scoring="average_precision",
            n_repeats=5, random_state=seed, n_jobs=-1,
        )
        ranked = sorted(zip(MODEL_FEATURES, perm.importances_mean), key=lambda t: -t[1])
        importance = {k: round(float(v), 4) for k, v in ranked[:10]}

    metrics = {
        "dataset": {
            "transactions": int(len(features)),
            "fraud_rate": round(float(features["is_fraud"].mean()), 4),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_window": [str(train["timestamp"].min()), str(train["timestamp"].max())],
            "test_window": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        },
        "models": results,
        "best_model": best_name,
        "feature_importance": importance,
    }

    # ---- persist ------------------------------------------------------------
    joblib.dump(
        {
            "model": fitted[best_name],
            "name": best_name,
            "threshold": results[best_name]["threshold"],
            "features": MODEL_FEATURES,
        },
        artifact_dir / "model.joblib",
    )
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FinSentinel fraud models")
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    args = parser.parse_args()

    metrics = train_pipeline(
        n_customers=args.customers,
        days=args.days,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        artifact_dir=args.artifacts,
        quick=args.quick,
    )

    print("\n=== FinSentinel training report ===")
    ds = metrics["dataset"]
    print(
        f"dataset: {ds['transactions']:,} txns | fraud {ds['fraud_rate']:.2%} | "
        f"train {ds['train_rows']:,} / test {ds['test_rows']:,}"
    )
    header = f"{'model':<22}{'PR-AUC':>8}{'ROC-AUC':>9}{'precision':>11}{'recall':>8}"
    print(header)
    for name, r in metrics["models"].items():
        print(
            f"{name:<22}{r['average_precision']:>8.3f}{r['roc_auc']:>9.3f}"
            f"{r['precision']:>11.3f}{r['recall']:>8.3f}"
        )
    print(f"\nbest model: {metrics['best_model']}  -> artifacts/model.joblib")


if __name__ == "__main__":
    main()
