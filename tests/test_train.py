"""Smoke tests for the training pipeline."""

from pathlib import Path

import joblib

from fin_sentinel.train import train_pipeline


def test_quick_pipeline_writes_artifacts(tmp_path):
    metrics = train_pipeline(artifact_dir=tmp_path, data_dir=tmp_path / "data", quick=True)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "data" / "transactions.csv").exists()

    assert metrics["best_model"] in metrics["models"]
    best = metrics["models"][metrics["best_model"]]
    assert best["average_precision"] > 0.0
    assert 0.0 <= best["threshold"] <= 1.0 or metrics["best_model"] == "isolation_forest"

    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["name"] == metrics["best_model"]
    assert "is_fraud" not in bundle["features"]
