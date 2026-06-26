"""
routes.py
All FastAPI route definitions. Imported by main.py.
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

import numpy as np
from fastapi.responses import Response
from app.pdf_report import generate_pdf_report
from src.api.schemas import (
    LoanApplication, PredictionResponse, ExplainRequest,
    ExplainResponse, HealthResponse, SHAPFactor,
    PDFGenerateRequest, WhatIfSweepResponse, SweepCurve,
    DemographicGroupAudit, FairnessAuditResponse,
    GlobalSHAPImportance, DemographicSHAPDetails,
)
from src.data.preprocessing import LoanPreprocessor, FINAL_FEATURES
from src.explain.shap_explainer import (
    get_shap_values_for_instance, get_top_factors,
)
from src.explain.llm_explainer import generate_llm_explanation

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
SAVED_MODELS_DIR = _HERE.parent.parent / 'saved_models'

router = APIRouter()

# ── Module-level artifact cache (loaded once at startup) ──────────────────────
_model = None
_preprocessor: LoanPreprocessor | None = None
_model_info: dict | None = None
_shap_explainer = None
_feature_names: list[str] = []


def load_artifacts() -> None:
    """Load all model artifacts at API startup. Raises on missing files."""
    global _model, _preprocessor, _model_info, _shap_explainer, _feature_names

    required = ['model.pkl', 'preprocessor.pkl', 'model_info.json']
    for f in required:
        if not (SAVED_MODELS_DIR / f).exists():
            raise FileNotFoundError(
                f"Required artifact '{f}' not found in {SAVED_MODELS_DIR}/. "
                "Run 'python src/models/train.py' first."
            )

    _model = joblib.load(SAVED_MODELS_DIR / 'model.pkl')
    _preprocessor = LoanPreprocessor.load(str(SAVED_MODELS_DIR / 'preprocessor.pkl'))

    with open(SAVED_MODELS_DIR / 'model_info.json') as fh:
        _model_info = json.load(fh)

    _feature_names = _model_info.get('feature_names', FINAL_FEATURES)

    # SHAP explainer (optional — don't block if missing)
    shap_path = SAVED_MODELS_DIR / 'shap_explainer.pkl'
    if shap_path.exists():
        try:
            _shap_explainer = joblib.load(shap_path)
        except Exception as e:
            print(f"WARNING: Could not load SHAP explainer: {e}")

    print("[OK] All artifacts loaded successfully.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _application_to_df(app: LoanApplication) -> pd.DataFrame:
    """Convert Pydantic model -> raw DataFrame row for preprocessing."""
    return pd.DataFrame([{
        'Gender': app.gender,
        'Marital_Status': app.marital_status,
        'Dependents': app.dependents,
        'Education_Level': app.education_level,
        'Employment_Status': app.employment_status,
        'Employer_Category': app.employer_category,
        'Age': app.age,
        'Applicant_Income': app.applicant_income,
        'Coapplicant_Income': app.coapplicant_income,
        'Loan_Amount': app.loan_amount,
        'Loan_Term': app.loan_term,
        'Existing_Loans': app.existing_loans,
        'DTI_Ratio': app.dti_ratio,
        'Savings': app.savings,
        'Collateral_Value': app.collateral_value,
        'Credit_Score': app.credit_score,
        'Property_Area': app.property_area,
        'Loan_Purpose': app.loan_purpose,
    }])


def _get_confidence(probability: float) -> str:
    if probability >= 0.75 or probability <= 0.25:
        return 'High'
    elif probability >= 0.62 or probability <= 0.38:
        return 'Medium'
    return 'Low'


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns model health status and performance metrics."""
    if _model is None or _model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Server may still be starting."
        )
    return HealthResponse(
        status='healthy',
        model_version=_model_info.get('model_version', 'unknown'),
        model_type=_model_info.get('model_type', 'unknown'),
        performance=_model_info.get('performance', {}),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication):
    """
    Predict loan approval with SHAP feature explanations.

    Returns:
    - **approved**: boolean decision
    - **probability**: approval probability (0.0–1.0)
    - **confidence**: High / Medium / Low
    - **shap_values**: per-feature SHAP contribution scores
    - **top_factors**: top positive and negative factors
    """
    if _model is None or _preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )

    try:
        raw_df = _application_to_df(application)
        X = _preprocessor.transform(raw_df)

        prob = float(_model.predict_proba(X)[0][1])
        approved = prob >= 0.5

        # SHAP values
        shap_dict: dict[str, float] = {}
        top_factors_list: list[SHAPFactor] = []

        if _shap_explainer is not None:
            try:
                shap_dict = get_shap_values_for_instance(
                    _shap_explainer, X, _feature_names
                )
                top_raw = get_top_factors(shap_dict)
                for item in top_raw['positive']:
                    top_factors_list.append(SHAPFactor(
                        feature=item['feature'],
                        value=item['value'],
                        direction='positive',
                    ))
                for item in top_raw['negative']:
                    top_factors_list.append(SHAPFactor(
                        feature=item['feature'],
                        value=item['value'],
                        direction='negative',
                    ))
            except Exception as shap_err:
                print(f"WARNING: SHAP computation failed (non-breaking): {shap_err}")

        return PredictionResponse(
            approved=approved,
            probability=round(prob, 4),
            confidence=_get_confidence(prob),
            decision_text="✅ Loan Approved" if approved else "❌ Loan Rejected",
            shap_values=shap_dict,
            top_factors=top_factors_list,
            model_version=_model_info.get('model_version', '1.0.0'),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(request: ExplainRequest):
    """
    Generate a plain-English LLM explanation for a prediction result.
    Uses Claude API (falls back to rule-based if ANTHROPIC_API_KEY is missing).
    """
    try:
        result = generate_llm_explanation(
            application=request.application,
            prediction=request.prediction,
        )
        return ExplainResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(e)}",
        )


@router.post("/whatif_sweep", response_model=WhatIfSweepResponse, tags=["What-If"])
async def whatif_sweep(application: LoanApplication):
    """
    Sweeps 6 key features to compute probability curves and find approval flip points.
    """
    if _model is None or _preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )

    try:
        # 1. Base prediction
        base_df = _application_to_df(application)
        base_X = _preprocessor.transform(base_df)
        base_prob = float(_model.predict_proba(base_X)[0][1])
        base_approved = base_prob >= 0.5

        # 2. Key features configurations to sweep
        sweep_configs = {
            "credit_score": {
                "label": "Credit Score",
                "vals": list(range(300, 901, 15)),
                "hint": "Higher is better. 750+ is typically strong.",
            },
            "dti_ratio": {
                "label": "Debt-to-Income Ratio",
                "vals": [round(float(v), 2) for v in np.arange(0.05, 0.96, 0.04)],
                "hint": "Lower is better. Aim for < 40%.",
            },
            "savings": {
                "label": "Savings Amount",
                "vals": list(range(0, 200_001, 5_000)),
                "hint": "Higher savings reduce lender risk.",
            },
            "loan_amount": {
                "label": "Loan Amount",
                "vals": list(range(5_000, 300_001, 5_000)),
                "hint": "Lower loan amounts are easier to approve.",
            },
            "coapplicant_income": {
                "label": "Co-applicant Income",
                "vals": list(range(0, 30_001, 1_000)),
                "hint": "Adding a co-applicant income strengthens the application.",
            },
            "existing_loans": {
                "label": "Existing Active Loans",
                "vals": list(range(0, 11, 1)),
                "hint": "Fewer existing loans improve approval chances.",
            },
        }

        # 3. Perform sweep
        curves = []
        base_dict = application.model_dump()

        for feat_key, config in sweep_configs.items():
            vals = config["vals"]
            probs = []
            flip_val = None
            flip_prob = None

            # Prepare batch inputs
            sweep_rows = []
            for v in vals:
                test_dict = {**base_dict, feat_key: v}
                # Create app model to run validators
                test_app = LoanApplication(**test_dict)
                sweep_rows.append(_application_to_df(test_app).iloc[0])

            sweep_df = pd.DataFrame(sweep_rows)
            X_sweep = _preprocessor.transform(sweep_df)
            probs_array = _model.predict_proba(X_sweep)[:, 1]

            for idx, prob_val in enumerate(probs_array):
                prob_val = float(prob_val)
                probs.append(round(prob_val, 4))

                is_approved = prob_val >= 0.5
                if not base_approved and is_approved and flip_val is None:
                    flip_val = vals[idx]
                    flip_prob = round(prob_val, 4)

            curves.append(SweepCurve(
                feature=feat_key,
                label=config["label"],
                vals=[float(v) for v in vals],
                probs=probs,
                current_val=float(base_dict[feat_key]),
                flip_val=float(flip_val) if flip_val is not None else None,
                flip_prob=flip_prob,
                hint=config["hint"],
            ))

        return WhatIfSweepResponse(
            base_probability=round(base_prob, 4),
            base_approved=base_approved,
            curves=curves,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"What-If computation failed: {str(e)}",
        )


@router.get("/fairness", response_model=FairnessAuditResponse, tags=["Fairness"])
async def fairness_audit():
    """
    Runs predictions on the entire dataset and returns metrics, DIR rates, SHAP rankings, and histogram bins.
    """
    if _model is None or _preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )

    try:
        data_path = SAVED_MODELS_DIR.parent / "data" / "loanData.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)
        df = df.dropna(subset=["Loan_Approved"]).reset_index(drop=True)

        drop_cols = ["Loan_Approved", "Applicant_ID"] if "Applicant_ID" in df.columns else ["Loan_Approved"]
        X_raw = df.drop(columns=drop_cols, errors="ignore")
        X = _preprocessor.transform(X_raw)

        probs = _model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["predicted_prob"] = probs
        df["predicted_approved"] = preds

        total_applicants = len(df)
        predicted_approved = int(preds.sum())
        approval_rate = float(predicted_approved / total_applicants * 100)
        roc_auc = float(_model_info.get("performance", {}).get("roc_auc", 0.988))

        # Demographic Group Audits
        group_cols = [
            ("Gender", "Gender"),
            ("Property_Area", "Property Area"),
            ("Employment_Status", "Employment Status"),
            ("Education_Level", "Education Level"),
        ]
        groups = []

        for col_name, display_label in group_cols:
            if col_name not in df.columns:
                continue

            agg = (
                df.groupby(col_name)["predicted_approved"]
                .agg(["mean", "count"])
                .reset_index()
            )
            agg["mean"] = agg["mean"].fillna(0.0)

            categories = agg[col_name].astype(str).tolist()
            rates = (agg["mean"] * 100).round(2).tolist()
            counts = agg["count"].astype(int).tolist()

            # Compute Disparate Impact Ratio (DIR)
            rates_fraction = agg["mean"].tolist()
            if len(rates_fraction) >= 2:
                ref = max(rates_fraction)
                minority = min(rates_fraction)
                ratio = minority / ref if ref > 0 else 1.0
            else:
                ratio = 1.0

            # Status classification
            if ratio >= 0.9:
                status_text = "Pass"
                color = "#3fb950"  # Green
            elif ratio >= 0.8:
                status_text = "Marginal"
                color = "#e3b341"  # Amber
            else:
                status_text = "Adverse Impact"
                color = "#f85149"  # Red

            groups.append(DemographicGroupAudit(
                feature=display_label,
                categories=categories,
                approval_rates=rates,
                counts=counts,
                dir_value=round(ratio, 4),
                dir_status=status_text,
                dir_color=color,
            ))

        # SHAP Importance
        global_shap = []
        demographic_shap = []
        shap_importance = {}

        if _shap_explainer is not None:
            try:
                sv = _shap_explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                mean_abs = np.abs(sv).mean(axis=0)
                shap_importance = dict(zip(_feature_names, mean_abs.tolist()))
            except Exception:
                pass

        if shap_importance:
            sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Map database feature names to display names if available
            from app.utils import FEATURE_DISPLAY
            
            for k, v in sorted_shap[:15]:
                disp_lbl = FEATURE_DISPLAY.get(k, k.replace("_", " "))
                global_shap.append(GlobalSHAPImportance(
                    feature=disp_lbl,
                    importance=round(v, 4),
                ))

            # Demographic feature ranks
            demographic_features = {"Gender", "Marital_Status", "Education_Level", "Age"}
            sorted_keys = [k for k, _ in sorted_shap]
            for feat in demographic_features:
                if feat in shap_importance:
                    rank = sorted_keys.index(feat) + 1
                    disp_lbl = FEATURE_DISPLAY.get(feat, feat)
                    demographic_shap.append(DemographicSHAPDetails(
                        feature=disp_lbl,
                        importance=round(shap_importance[feat], 4),
                        rank=rank,
                    ))

        # Histogram Bins (0.0 to 1.0 with width 0.05)
        bins = [round(float(x), 2) for x in np.arange(0.0, 1.01, 0.05)]
        approved_probs = df[df["predicted_approved"] == 1]["predicted_prob"].tolist()
        rejected_probs = df[df["predicted_approved"] == 0]["predicted_prob"].tolist()

        approved_counts, _ = np.histogram(approved_probs, bins=bins)
        rejected_counts, _ = np.histogram(rejected_probs, bins=bins)

        return FairnessAuditResponse(
            total_applicants=total_applicants,
            predicted_approved=predicted_approved,
            approval_rate=round(approval_rate, 2),
            roc_auc=roc_auc,
            groups=groups,
            global_shap=global_shap,
            demographic_shap=demographic_shap,
            histogram_bins=bins,
            histogram_approved=approved_counts.tolist(),
            histogram_rejected=rejected_counts.tolist(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fairness computation failed: {str(e)}",
        )


@router.post("/pdf", tags=["Report"])
async def download_pdf_report(request: PDFGenerateRequest):
    """
    Generate and download the loan assessment PDF report.
    """
    try:
        payload_dict = request.application.model_dump()
        result_dict = request.prediction.model_dump()
        explanation_dict = request.explanation.model_dump()

        pdf_bytes = generate_pdf_report(payload_dict, result_dict, explanation_dict)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=creditwise_assessment_{'approved' if result_dict['approved'] else 'rejected'}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF report generation failed: {str(e)}",
        )

