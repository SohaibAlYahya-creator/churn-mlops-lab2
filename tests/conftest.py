from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.train.train import train

DATA = Path("data/churn_v1.csv")

SAMPLE_CUSTOMER = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    """Train once per test run into a temporary folder (never touches models/)."""
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    train(DATA, path)
    return path


@pytest.fixture()
def client(model_file, monkeypatch):
    """A test client whose app loads the temporary model."""
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    with TestClient(app) as test_client:
        yield test_client
