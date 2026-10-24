import pytest

from tests.conftest import SAMPLE_CUSTOMER


def test_health_contract(client):
    """API contract: /health answers 200 with the fields clients rely on."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "model_version" in body


@pytest.mark.parametrize(
    "change",
    [
        {"contract": "weekly"},
        {"internet_service": "satellite"},
        {"tenure_months": -5},
        {"monthly_charges": "expensive"},
        {"support_calls": 1000},
    ],
)
def test_input_schema_rejects_bad_values(client, change):
    """Input schema: invalid values never reach the model (HTTP 422)."""
    response = client.post("/predict", json={**SAMPLE_CUSTOMER, **change})
    assert response.status_code == 422


def test_input_schema_rejects_missing_field(client):
    payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "support_calls"}
    assert client.post("/predict", json=payload).status_code == 422


def test_edge_case_boundary_values(client):
    """Edge case: the smallest and largest allowed values still work."""
    smallest = {
        **SAMPLE_CUSTOMER,
        "tenure_months": 0,
        "monthly_charges": 0,
        "support_calls": 0,
        "internet_service": "none",
    }
    largest = {
        **SAMPLE_CUSTOMER,
        "tenure_months": 120,
        "monthly_charges": 1000,
        "support_calls": 100,
        "contract": "two-year",
    }
    for payload in (smallest, largest):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_deterministic_prediction(client):
    """Same input, same output: matches the Lab 1 value for the sample customer."""
    first = client.post("/predict", json=SAMPLE_CUSTOMER).json()
    second = client.post("/predict", json=SAMPLE_CUSTOMER).json()
    assert first == second
    assert first["churn_probability"] == pytest.approx(0.866, abs=0.001)
