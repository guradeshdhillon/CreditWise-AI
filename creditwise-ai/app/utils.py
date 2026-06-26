"""
utils.py
Shared standalone inference utilities for all CreditWise AI Streamlit pages.

Provides direct model loading (bypasses FastAPI) so the app works both
locally (with the API) and on Streamlit Cloud (without a running server).
"""

import sys
import json
import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.preprocessing import LoanPreprocessor, FINAL_FEATURES
from src.explain.shap_explainer import get_shap_values_for_instance, get_top_factors
from src.explain.llm_explainer import generate_llm_explanation

# ── Human-readable feature display names ─────────────────────────────────────
FEATURE_DISPLAY = {
    "Gender": "Gender",
    "Marital_Status": "Marital Status",
    "Dependents": "Number of Dependents",
    "Education_Level": "Education Level",
    "Employment_Status": "Employment Status",
    "Employer_Category": "Employer Category",
    "Property_Area": "Property Area",
    "Loan_Purpose": "Loan Purpose",
    "Applicant_Income": "Monthly Income (Applicant)",
    "Coapplicant_Income": "Monthly Income (Co-applicant)",
    "Age": "Applicant Age",
    "Credit_Score": "Credit Score",
    "Existing_Loans": "Existing Active Loans",
    "DTI_Ratio": "Debt-to-Income Ratio",
    "Savings": "Savings Amount",
    "Collateral_Value": "Collateral Value",
    "Loan_Amount": "Loan Amount",
    "Loan_Term": "Loan Term (months)",
    "Total_Income": "Total Household Income",
    "EMI": "Monthly EMI",
    "Balance_Income": "Balance Income After EMI",
    "Income_to_Loan_Ratio": "Income-to-Loan Ratio",
    "Savings_to_Loan_Ratio": "Savings-to-Loan Ratio",
    "Collateral_to_Loan_Ratio": "Collateral-to-Loan Ratio",
    "LoanAmount_log": "Loan Amount (log-scaled)",
    "Total_Income_log": "Total Income (log-scaled)",
    "Credit_Score_norm": "Credit Score (normalised)",
}


# ── Cached artifact loader ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    """
    Load and cache model + preprocessor + SHAP explainer.
    Called once per Streamlit session; subsequent calls use the cache.
    """
    model = joblib.load(_ROOT / "saved_models" / "model.pkl")
    preprocessor = LoanPreprocessor.load(str(_ROOT / "saved_models" / "preprocessor.pkl"))

    with open(_ROOT / "saved_models" / "model_info.json") as fh:
        info = json.load(fh)

    shap_path = _ROOT / "saved_models" / "shap_explainer.pkl"
    shap_explainer = joblib.load(shap_path) if shap_path.exists() else None

    return model, preprocessor, info, shap_explainer


# ── Payload helpers ───────────────────────────────────────────────────────────

def payload_to_df(payload: dict) -> pd.DataFrame:
    """Convert a flat payload dict to the raw DataFrame format the preprocessor expects."""
    return pd.DataFrame([{
        "Gender":             payload["gender"],
        "Marital_Status":     payload["marital_status"],
        "Dependents":         int(payload["dependents"]),
        "Education_Level":    payload["education_level"],
        "Employment_Status":  payload["employment_status"],
        "Employer_Category":  payload["employer_category"],
        "Age":                int(payload["age"]),
        "Applicant_Income":   float(payload["applicant_income"]),
        "Coapplicant_Income": float(payload["coapplicant_income"]),
        "Loan_Amount":        float(payload["loan_amount"]),
        "Loan_Term":          float(payload["loan_term"]),
        "Existing_Loans":     int(payload["existing_loans"]),
        "DTI_Ratio":          float(payload["dti_ratio"]),
        "Savings":            float(payload["savings"]),
        "Collateral_Value":   float(payload["collateral_value"]),
        "Credit_Score":       int(payload["credit_score"]),
        "Property_Area":      payload["property_area"],
        "Loan_Purpose":       payload["loan_purpose"],
    }])


def _get_confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    elif prob >= 0.62 or prob <= 0.38:
        return "Medium"
    return "Low"


# ── Core standalone inference ─────────────────────────────────────────────────

def standalone_predict(payload: dict) -> dict:
    """
    Run prediction directly (no FastAPI required).
    Returns a result dict identical in shape to the /predict API response.
    """
    model, preprocessor, info, shap_explainer = load_artifacts()
    feature_names = info.get("feature_names", FINAL_FEATURES)

    raw_df = payload_to_df(payload)
    X = preprocessor.transform(raw_df)

    prob = float(model.predict_proba(X)[0][1])
    approved = prob >= 0.5

    shap_dict: dict = {}
    top_factors_list: list = []
    if shap_explainer is not None:
        try:
            shap_dict = get_shap_values_for_instance(shap_explainer, X, feature_names)
            top_raw = get_top_factors(shap_dict)
            for item in top_raw["positive"]:
                top_factors_list.append({
                    "feature": item["feature"], "value": item["value"], "direction": "positive"
                })
            for item in top_raw["negative"]:
                top_factors_list.append({
                    "feature": item["feature"], "value": item["value"], "direction": "negative"
                })
        except Exception:
            pass

    return {
        "approved":      approved,
        "probability":   round(prob, 4),
        "confidence":    _get_confidence(prob),
        "decision_text": "✅ Loan Approved" if approved else "❌ Loan Rejected",
        "shap_values":   shap_dict,
        "top_factors":   top_factors_list,
        "model_version": info.get("model_version", "1.0.0"),
    }


def standalone_explain(payload: dict, result: dict) -> dict:
    """
    Generate an LLM explanation directly (no FastAPI required).
    Falls back to rule-based if ANTHROPIC_API_KEY is not set.
    """
    # Streamlit Cloud: read key from st.secrets
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("credentials", {}).get("ANTHROPIC_API_KEY", "")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
        except Exception:
            pass

    return generate_llm_explanation(application=payload, prediction=result)
