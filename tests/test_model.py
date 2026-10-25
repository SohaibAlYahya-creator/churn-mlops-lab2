import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.train.train import DEFAULT_DATA, train


def test_model_loads_and_predicts(model_file):
    """The saved model file loads and returns a probability."""
    model = joblib.load(model_file)
    row = pd.DataFrame(
        [
            {
                "tenure_months": 5,
                "monthly_charges": 89.9,
                "contract": "month-to-month",
                "internet_service": "fiber",
                "support_calls": 3,
            }
        ]
    )
    proba = model.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_service_fails_loudly_without_model(monkeypatch, tmp_path):
    """Failure case: no model file means the service must not start silently."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_training_is_reproducible(tmp_path):
    """Same data and same code give the same metrics (the Lab 1 numbers)."""
    metrics = train(DEFAULT_DATA, tmp_path / "m.joblib")
    assert metrics == {"f1": 0.512, "auc": 0.783}


def test_training_rejects_renamed_column(tmp_path):
    """Lab 1 Twist 1: a renamed column must raise a clear error."""
    df = pd.read_csv(DEFAULT_DATA).rename(columns={"monthly_charges": "monthly_fee"})
    bad_csv = tmp_path / "churn_v2_like.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="monthly_charges"):
        train(bad_csv, tmp_path / "m.joblib")
