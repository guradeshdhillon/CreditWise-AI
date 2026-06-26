"""
llm_explainer.py
Generates plain-English loan decision explanations using Claude API.
Requires ANTHROPIC_API_KEY in .env
Falls back to rule-based explanation if API key is missing.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# Human-readable display names for features
FEATURE_DISPLAY_NAMES = {
    'Gender': 'Gender',
    'Marital_Status': 'Marital Status',
    'Dependents': 'Number of Dependents',
    'Education_Level': 'Education Level',
    'Employment_Status': 'Employment Status',
    'Employer_Category': 'Employer Category',
    'Property_Area': 'Property Area',
    'Applicant_Income': 'Applicant Monthly Income',
    'Coapplicant_Income': 'Co-applicant Monthly Income',
    'Age': 'Applicant Age',
    'Credit_Score': 'Credit Score',
    'Existing_Loans': 'Number of Existing Loans',
    'DTI_Ratio': 'Debt-to-Income Ratio',
    'Savings': 'Savings Amount',
    'Collateral_Value': 'Collateral Value',
    'Loan_Amount': 'Loan Amount Requested',
    'Loan_Term': 'Loan Term (months)',
    'Total_Income': 'Total Household Income',
    'EMI': 'Monthly EMI Estimate',
    'Balance_Income': 'Balance Income After EMI',
    'Income_to_Loan_Ratio': 'Income-to-Loan Ratio',
    'Savings_to_Loan_Ratio': 'Savings-to-Loan Ratio',
    'Collateral_to_Loan_Ratio': 'Collateral-to-Loan Ratio',
    'LoanAmount_log': 'Loan Amount (scaled)',
    'Total_Income_log': 'Total Income (scaled)',
    'Credit_Score_norm': 'Normalised Credit Score',
    'Loan_Purpose': 'Loan Purpose',
}


def _build_shap_summary(shap_values: dict) -> str:
    """Convert SHAP dict to concise readable lines for the LLM prompt."""
    if not shap_values:
        return "SHAP feature analysis not available for this prediction."
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    lines = []
    for feature, value in sorted_items:
        display = FEATURE_DISPLAY_NAMES.get(feature, feature)
        direction = "↑ supports approval" if value > 0 else "↓ reduces approval chances"
        lines.append(f"  - {display}: {value:+.3f} ({direction})")
    return "\n".join(lines)


def _build_prompt(application: dict, prediction: dict) -> str:
    """Build a structured prompt for the Claude API."""
    decision = "APPROVED ✓" if prediction.get('approved') else "REJECTED ✗"
    prob = prediction.get('probability', 0.5)
    conf = prediction.get('confidence', 'Medium')
    shap_summary = _build_shap_summary(prediction.get('shap_values', {}))

    return f"""You are a senior loan officer at a bank. A machine learning model has analyzed a loan application and produced the following result. Write a clear, empathetic, and professional explanation for the applicant.

== LOAN DECISION ==
Decision: {decision}
Approval Probability: {prob:.1%}
Confidence Level: {conf}

== APPLICATION DETAILS ==
Credit Score: {application.get('credit_score', 'N/A')}
Monthly Income (Applicant): ₹{application.get('applicant_income', 0):,.0f}
Monthly Income (Co-applicant): ₹{application.get('coapplicant_income', 0):,.0f}
Loan Amount Requested: ₹{application.get('loan_amount', 0):,.0f}
Loan Term: {application.get('loan_term', 0)} months
DTI Ratio: {application.get('dti_ratio', 0):.1%}
Existing Loans: {application.get('existing_loans', 0)}
Savings: ₹{application.get('savings', 0):,.0f}
Collateral Value: ₹{application.get('collateral_value', 0):,.0f}
Employment: {application.get('employment_status', 'N/A')} at {application.get('employer_category', 'N/A')}
Education: {application.get('education_level', 'N/A')}
Property Area: {application.get('property_area', 'N/A')}

== AI FEATURE IMPORTANCE (SHAP Values) ==
{shap_summary}

Instructions:
1. Write 2-3 sentences explaining the decision in plain English (no jargon, no mention of "SHAP" or "XGBoost").
2. Identify the single most impactful factor.
3. Provide exactly 3 specific, actionable improvement tips if rejected, or 2 tips to maintain good standing if approved.

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "explanation": "Your 2-3 sentence plain English explanation here.",
  "key_factor": "The single most important factor in one sentence.",
  "improvement_tips": ["Tip 1", "Tip 2", "Tip 3"]
}}"""


def _rule_based_explanation(application: dict, prediction: dict) -> dict:
    """Fallback rule-based explanation when Claude API is unavailable."""
    approved = prediction.get('approved', False)
    credit_score = application.get('credit_score', 650)
    dti = application.get('dti_ratio', 0.35)
    shap_vals = prediction.get('shap_values', {})

    # Find key factor from SHAP
    if shap_vals:
        key_feature = max(shap_vals, key=lambda k: abs(shap_vals[k]))
        key_display = FEATURE_DISPLAY_NAMES.get(key_feature, key_feature)
    else:
        key_display = "Credit Score" if credit_score < 650 else "Debt-to-Income Ratio"

    if approved:
        explanation = (
            f"Congratulations! Your loan application has been approved with "
            f"{prediction.get('probability', 0.5):.0%} confidence. "
            f"Your financial profile demonstrates strong creditworthiness and repayment capacity."
        )
        key_factor = f"Your {key_display} was the most significant positive factor in this decision."
        tips = [
            "Continue making timely payments on all existing loans to maintain your credit score.",
            "Keep your debt-to-income ratio below 40% for future loan eligibility.",
        ]
    else:
        explanation = (
            f"Unfortunately, your loan application has been declined at this time. "
            f"The model assessed your application with a {prediction.get('probability', 0.5):.0%} approval probability. "
            f"This decision is based on factors including your credit profile and financial ratios."
        )
        key_factor = f"Your {key_display} was the most significant factor influencing this outcome."
        tips = [
            f"Improve your credit score — currently {credit_score} — by paying bills on time and reducing credit card balances.",
            f"Reduce your debt-to-income ratio (currently {dti:.0%}) by paying down existing loans before applying.",
            "Consider applying with a co-applicant who has a strong income, or offering additional collateral.",
        ]

    return {
        "explanation": explanation,
        "key_factor": key_factor,
        "improvement_tips": tips,
    }


def generate_llm_explanation(application, prediction) -> dict:
    """
    Generate a plain-English explanation using Claude API.
    Falls back gracefully to rule-based if API key is missing or call fails.

    Args:
        application: LoanApplication pydantic model (or dict)
        prediction:  PredictionResponse pydantic model (or dict)
    Returns:
        dict with keys: explanation, key_factor, improvement_tips
    """
    # Convert pydantic models to dicts if needed
    if hasattr(application, 'model_dump'):
        app_dict = application.model_dump()
    elif hasattr(application, 'dict'):
        app_dict = application.dict()
    else:
        app_dict = dict(application)

    if hasattr(prediction, 'model_dump'):
        pred_dict = prediction.model_dump()
    elif hasattr(prediction, 'dict'):
        pred_dict = prediction.dict()
    else:
        pred_dict = dict(prediction)

    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("INFO: ANTHROPIC_API_KEY not set -- using rule-based fallback.")
        return _rule_based_explanation(app_dict, pred_dict)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = _build_prompt(app_dict, pred_dict)

        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = message.content[0].text.strip()

        # Parse JSON response
        result = json.loads(raw_text)

        # Validate required keys
        for key in ('explanation', 'key_factor', 'improvement_tips'):
            if key not in result:
                raise ValueError(f"Missing key '{key}' in LLM response")

        return result

    except json.JSONDecodeError as e:
        print(f"WARNING: LLM returned invalid JSON: {e}. Using fallback.")
        return _rule_based_explanation(app_dict, pred_dict)
    except Exception as e:
        print(f"WARNING: Claude API call failed: {e}. Using fallback.")
        return _rule_based_explanation(app_dict, pred_dict)
