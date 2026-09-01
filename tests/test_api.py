"""Tests for the FastAPI scoring service."""

import pytest
from fastapi.testclient import TestClient

from fin_sentinel.serve import app
from fin_sentinel.train import train_pipeline


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    return tmp_path_factory.mktemp("artifacts")


@pytest.fixture(scope="module")
def client(trained_artifacts, monkeypatch_module):
    train_pipeline(artifact_dir=trained_artifacts, data_dir=trained_artifacts / "data", quick=True)
    monkeypatch_module.setenv("FIN_SENTINEL_MODEL", str(trained_artifacts / "model.joblib"))
    monkeypatch_module.setenv("FIN_SENTINEL_ARTIFACTS", str(trained_artifacts))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    import pytest as _pytest

    return _pytest.MonkeyPatch()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_score_returns_probability_and_band(client):
    payload = {
        "amount": 2499.0,
        "category": "electronics",
        "hour": 3,
        "day_of_week": 6,
        "user_mean_prior": 42.0,
        "minutes_since_prior_txn": 2.0,
        "txn_count_1h": 6,
        "txn_count_24h": 9,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["recommended_action"]


def test_score_low_risk_transaction(client):
    payload = {
        "amount": 38.5,
        "category": "groceries",
        "hour": 14,
        "day_of_week": 2,
        "user_mean_prior": 40.0,
        "minutes_since_prior_txn": 900.0,
        "txn_count_1h": 0,
        "txn_count_24h": 1,
    }
    body = client.post("/score", json=payload).json()
    assert body["fraud_probability"] < 0.5


def test_invalid_category_rejected(client):
    payload = {"amount": 10.0, "category": "cryptocurrency", "hour": 3, "user_mean_prior": 40.0}
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "best_model" in response.json()
