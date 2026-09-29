import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import CustomerFeatures

VALID = CustomerFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "threshold" in r.json()


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    # Intentionally wrong bound to prove CI catches a broken test
    assert 0.0 <= body["churn_probability"] <= 0.01


def test_predict_missing_total_charges(client):
    payload = copy.deepcopy(VALID)
    payload["TotalCharges"] = None
    assert client.post("/predict", json=payload).status_code == 200


def test_invalid_contract_rejected(client):
    payload = copy.deepcopy(VALID)
    payload["Contract"] = "Weekly"
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_tenure_rejected(client):
    payload = copy.deepcopy(VALID)
    payload["tenure"] = -5
    assert client.post("/predict", json=payload).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"customers": [VALID, VALID]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
