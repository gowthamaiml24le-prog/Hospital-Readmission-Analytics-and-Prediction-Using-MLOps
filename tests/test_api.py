"""
Integration tests for FastAPI inference endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model_name" in data
    assert "metrics_summary" in data

def test_predict_endpoint(client):
    payload = {
        "gender": "Female",
        "age_midpoint": 75,
        "is_elderly": 1,
        "length_of_stay": 5,
        "prior_inpatient_admissions": 2,
        "total_prior_visits": 3,
        "num_medications": 16,
        "diagnosis_count": 8,
        "admission_category": "Emergency",
        "discharge_category": "Transferred / Skilled Facility",
        "diag_1_category": "Circulatory"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_risk_probability" in data
    assert 0.0 <= data["readmission_risk_probability"] <= 1.0
    assert data["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "clinical_recommendation" in data

def test_explain_endpoint(client):
    payload = {
        "gender": "Male",
        "age_midpoint": 65,
        "is_elderly": 1,
        "length_of_stay": 4,
        "prior_inpatient_admissions": 1,
        "total_prior_visits": 2,
        "num_medications": 14,
        "diag_1_category": "Diabetes",
        "discharge_category": "Discharged to Home"
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_probability" in data
    assert "top_contributing_risk_factors" in data
    assert len(data["top_contributing_risk_factors"]) > 0
