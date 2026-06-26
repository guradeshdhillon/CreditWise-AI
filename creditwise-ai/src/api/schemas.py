"""
schemas.py
Pydantic v2 models for request validation and response formatting.
Adapted to the actual loanData.csv 20-column dataset.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class LoanApplication(BaseModel):
    """Input schema for a loan prediction request."""
    model_config = ConfigDict(str_strip_whitespace=True)

    gender: Literal['Male', 'Female'] = Field(..., description="Applicant gender")
    marital_status: Literal['Married', 'Single'] = Field(..., description="Marital status")
    dependents: int = Field(..., ge=0, le=10, description="Number of dependents (0–10)")
    education_level: Literal['Graduate', 'Not Graduate'] = Field(..., description="Highest education")
    employment_status: Literal['Salaried', 'Self-employed', 'Contract', 'Unemployed'] = Field(
        ..., description="Current employment type"
    )
    employer_category: Literal['Private', 'Government', 'MNC', 'Business', 'Unemployed'] = Field(
        ..., description="Employer type"
    )
    age: int = Field(..., ge=18, le=85, description="Applicant age in years")
    applicant_income: float = Field(..., gt=0, description="Monthly applicant income (₹)")
    coapplicant_income: float = Field(..., ge=0, description="Monthly co-applicant income (₹)")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount (₹)")
    loan_term: float = Field(..., gt=0, description="Loan repayment term in months")
    existing_loans: int = Field(..., ge=0, le=20, description="Number of existing active loans")
    dti_ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-Income ratio (0.0–1.0)")
    savings: float = Field(..., ge=0, description="Total savings amount (₹)")
    collateral_value: float = Field(..., ge=0, description="Collateral/asset value (₹)")
    credit_score: int = Field(..., ge=300, le=900, description="Credit score (300–900)")
    property_area: Literal['Urban', 'Semiurban', 'Rural'] = Field(..., description="Property location")
    loan_purpose: Literal['Home', 'Car', 'Personal', 'Business', 'Education'] = Field(
        ..., description="Purpose of the loan"
    )

    @field_validator('applicant_income', 'coapplicant_income', 'loan_amount', 'savings', 'collateral_value')
    @classmethod
    def check_reasonable_values(cls, v: float, info) -> float:
        if v > 100_000_000:
            raise ValueError(f"{info.field_name} value {v} seems unreasonably large.")
        return v

    @field_validator('loan_term')
    @classmethod
    def check_loan_term(cls, v: float) -> float:
        if v > 600:
            raise ValueError(f"Loan term {v} months exceeds maximum of 600 months (50 years).")
        return v


class SHAPFactor(BaseModel):
    feature: str
    value: float
    direction: Literal['positive', 'negative']


class PredictionResponse(BaseModel):
    approved: bool
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: Literal['High', 'Medium', 'Low']
    decision_text: str
    shap_values: dict[str, float]
    top_factors: list[SHAPFactor]
    model_version: str


class ExplainRequest(BaseModel):
    """Combined request for LLM explanation."""
    application: LoanApplication
    prediction: PredictionResponse


class ExplainResponse(BaseModel):
    explanation: str
    key_factor: str
    improvement_tips: list[str]


class HealthResponse(BaseModel):
    status: Literal['healthy', 'degraded']
    model_version: str
    model_type: str
    performance: dict[str, float]


# ── Added schemas for upgraded frontend ───────────────────────────────────────

class PDFGenerateRequest(BaseModel):
    application: LoanApplication
    prediction: PredictionResponse
    explanation: ExplainResponse


class SweepCurve(BaseModel):
    feature: str
    label: str
    vals: list[float]
    probs: list[float]
    current_val: float
    flip_val: float | None = None
    flip_prob: float | None = None
    hint: str


class WhatIfSweepResponse(BaseModel):
    base_probability: float
    base_approved: bool
    curves: list[SweepCurve]


class DemographicGroupAudit(BaseModel):
    feature: str
    categories: list[str]
    approval_rates: list[float]
    counts: list[int]
    dir_value: float
    dir_status: str
    dir_color: str


class GlobalSHAPImportance(BaseModel):
    feature: str
    importance: float


class DemographicSHAPDetails(BaseModel):
    feature: str
    importance: float
    rank: int


class FairnessAuditResponse(BaseModel):
    total_applicants: int
    predicted_approved: int
    approval_rate: float
    roc_auc: float
    groups: list[DemographicGroupAudit]
    global_shap: list[GlobalSHAPImportance]
    demographic_shap: list[DemographicSHAPDetails]
    histogram_bins: list[float]
    histogram_approved: list[int]
    histogram_rejected: list[int]

