"""API tests with TestClient: /health, /predict and the client's two rules."""

from fastapi.testclient import TestClient

from app.main import app

VALID = {
    "distance_km": 1200, "weight_kg": 12.5, "hub_backlog": 2000, "pin_code_tier": 2,
    "declared_value_rs": 15000, "transfers": 3, "service": "express", "destination": "Tier-2",
}


def test_health_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_number_for_a_valid_request():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_declared_value_above_35000_is_rejected_with_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID, "declared_value_rs": 35001})
    assert response.status_code == 422


def test_negative_hub_backlog_is_rejected_with_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID, "hub_backlog": -1})
    assert response.status_code == 422


def test_spelling_variant_service_is_accepted():
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID, "service": "Exp."})
    assert response.status_code == 200
