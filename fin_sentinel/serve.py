"""Production-style scoring API for FinSentinel (FastAPI).

Clients send *raw* transaction context; the service recomputes the model
features server-side and returns a fraud probability, a risk band and an
recommended action. Run with::

    uvicorn fin_sentinel.serve:app --reload

Then::

    curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \\
      -d '{"amount": 2499.0, "category": "electronics", "hour": 3,
           "day_of_week": 6, "user_mean_prior": 42.0,
           "minutes_since_prior_txn": 2.0, "txn_count_1h": 6,
           "txn_count_24h": 9}'

The model path defaults to ``artifacts/model.joblib`` and can be overridden
with the ``FIN_SENTINEL_MODEL`` environment variable.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CATEGORIES, MODEL_FEATURES, RISK_BANDS, RECOMMENDATIONS

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


class ScoreRequest(BaseModel):
    """Raw transaction context as available at authorization time."""

    amount: float = Field(..., gt=0, description="Transaction amount")
    category: str = Field(..., description=f"One of: {', '.join(CATEGORIES)}")
    hour: int = Field(..., ge=0, le=23, description="Local hour of the transaction")
    day_of_week: int = Field(6, ge=0, le=6, description="0=Monday .. 6=Sunday")
    user_mean_prior: float = Field(
        ..., gt=0, description="Customer's mean spend on *prior* transactions"
    )
    minutes_since_prior_txn: float = Field(
        7 * 24 * 60.0, ge=0, description="Minutes since the customer's previous transaction"
    )
    txn_count_1h: float = Field(0.0, ge=0, description="Customer transactions in the last hour")
    txn_count_24h: float = Field(0.0, ge=0, description="Customer transactions in the last 24h")


class ScoreResponse(BaseModel):
    fraud_probability: float
    risk_band: str
    recommended_action: str
    model: str
    threshold: float


@lru_cache(maxsize=4)
def _load_bundle(path: str) -> dict:
    return joblib.load(path)


def get_bundle() -> dict:
    path = os.environ.get("FIN_SENTINEL_MODEL", str(DEFAULT_MODEL_PATH))
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. Train first: python -m fin_sentinel.train"
        )
    return _load_bundle(path)


def _risk_band(probability: float) -> str:
    for low, high, band in RISK_BANDS:
        if low <= probability < high:
            return band
    return RISK_BANDS[-1][2]


def _to_model_features(req: ScoreRequest) -> list[float]:
    """Mirror of ``features.engineer_features`` for a single request.

    The velocity features (``txn_count_1h`` / ``txn_count_24h``) and the
    recency feature arrive from the request, because in production they come
    from a feature store / stream counter rather than a batch table.
    """
    row = {
        "amount": req.amount,
        "log_amount": float(np.log1p(req.amount)),
        "hour_sin": float(np.sin(2 * np.pi * req.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * req.hour / 24.0)),
        "is_night": int(req.hour < 6),
        "is_weekend": int(req.day_of_week >= 5),
        "amount_vs_user_mean": req.amount / req.user_mean_prior,
        "minutes_since_prior_txn": req.minutes_since_prior_txn,
        "txn_count_1h": req.txn_count_1h,
        "txn_count_24h": req.txn_count_24h,
        "category_risk": 0.0 if req.category not in CATEGORIES else None,
    }
    # category risk comes from the static domain map
    from .config import CATEGORY_RISK

    row["category_risk"] = float(CATEGORY_RISK.get(req.category, 0.0))
    return [row[f] for f in MODEL_FEATURES]


app = FastAPI(
    title="FinSentinel Scoring API",
    description="Fraud probability scoring for card transactions. "
    "Built by Kaamchor Solutions.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
        return {"status": "ok", "model_loaded": True, "model": bundle["name"]}
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.get("/metadata")
def metadata() -> dict:
    path = Path(os.environ.get("FIN_SENTINEL_ARTIFACTS", str(DEFAULT_MODEL_PATH.parent)))
    metrics_file = path / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found -- train first")
    with open(metrics_file, encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if req.category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"category must be one of: {', '.join(CATEGORIES)}")
    bundle = get_bundle()
    features = _to_model_features(req)
    model = bundle["model"]
    X = pd.DataFrame([features], columns=bundle["features"])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0, 1])
    else:  # isolation forest: map the anomaly score linearly onto (0, 1)
        raw = float(-model.decision_function(X)[0])  # higher = more anomalous
        probability = float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))
    band = _risk_band(probability)
    return ScoreResponse(
        fraud_probability=round(probability, 4),
        risk_band=band,
        recommended_action=RECOMMENDATIONS[band],
        model=bundle["name"],
        threshold=bundle["threshold"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
