"""
test_api.py
Integration tests for FastAPI endpoints.
Skips automatically if model artifacts are not trained yet.
Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Auto-skip if artifacts not trained yet
if not (Path(__file__).resolve().parent.parent / 'saved_models' / 'model.pkl').exists():
    pytest.skip(
        "Model artifacts not found. Run 'python src/models/train.py' first.",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient
from src.api.main import app

# Use context manager so the lifespan event (load_artifacts) fires before tests
_client_ctx = TestClient(app, raise_server_exceptions=True)
_client_ctx.__enter__()
client = _client_ctx

VALID_APPLICATION = {
    "gender": "Male",
    "marital_status": "Married",
    "dependents": 1,
    "education_level": "Graduate",
    "employment_status": "Salaried",
    "employer_category": "Private",
    "age": 35,
    "applicant_income": 10000,
    "coapplicant_income": 2000,
    "loan_amount": 20000,
    "loan_term": 60,
    "existing_loans": 1,
    "dti_ratio": 0.30,
    "savings": 15000,
    "collateral_value": 40000,
    "credit_score": 720,
    "property_area": "Urban",
    "loan_purpose": "Home",
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "performance" in data


def test_predict_valid():
    r = client.post("/api/v1/predict", json=VALID_APPLICATION)
    assert r.status_code == 200
    data = r.json()
    assert "approved" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["confidence"] in ["High", "Medium", "Low"]
    assert "shap_values" in data
    assert "top_factors" in data


def test_predict_bad_credit():
    """Low credit score should return valid (but likely rejected) prediction."""
    app = {**VALID_APPLICATION, "credit_score": 350}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["approved"], bool)


def test_predict_high_dti():
    """Very high DTI should push toward rejection."""
    app = {**VALID_APPLICATION, "dti_ratio": 0.95}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 200


def test_predict_invalid_gender():
    """Invalid enum value should return 422 Unprocessable Entity."""
    app = {**VALID_APPLICATION, "gender": "Unknown"}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 422


def test_predict_negative_income():
    """Non-positive income should return 422."""
    app = {**VALID_APPLICATION, "applicant_income": -500}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 422


def test_predict_zero_loan():
    """Zero loan amount should return 422."""
    app = {**VALID_APPLICATION, "loan_amount": 0}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 422


def test_predict_invalid_credit_score_too_low():
    """Credit score below 300 should return 422."""
    app = {**VALID_APPLICATION, "credit_score": 200}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 422


def test_predict_invalid_dti_out_of_range():
    """DTI > 1.0 should return 422."""
    app = {**VALID_APPLICATION, "dti_ratio": 1.5}
    r = client.post("/api/v1/predict", json=app)
    assert r.status_code == 422


def test_whatif_sweep():
    r = client.post("/api/v1/whatif_sweep", json=VALID_APPLICATION)
    assert r.status_code == 200
    data = r.json()
    assert "base_probability" in data
    assert "base_approved" in data
    assert "curves" in data
    assert len(data["curves"]) == 6
    for curve in data["curves"]:
        assert "feature" in curve
        assert "vals" in curve
        assert "probs" in curve
        assert len(curve["vals"]) == len(curve["probs"])


def test_fairness():
    r = client.get("/api/v1/fairness")
    assert r.status_code == 200
    data = r.json()
    assert "total_applicants" in data
    assert "predicted_approved" in data
    assert "approval_rate" in data
    assert "groups" in data
    assert len(data["groups"]) > 0
    assert "global_shap" in data
    assert "histogram_bins" in data


def test_pdf():
    pred_res = client.post("/api/v1/predict", json=VALID_APPLICATION).json()
    explain_res = client.post("/api/v1/explain", json={
        "application": VALID_APPLICATION,
        "prediction": pred_res
    }).json()

    r = client.post("/api/v1/pdf", json={
        "application": VALID_APPLICATION,
        "prediction": pred_res,
        "explanation": explain_res
    })
    assert r.status_code == 200
    assert "application/pdf" in r.headers.get("content-type", "")
    assert len(r.content) > 0

