"""
pdf_report.py
Generates a professional downloadable PDF loan assessment report.
Uses fpdf2 with embedded matplotlib SHAP chart.
"""

import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

# ── Feature display names ─────────────────────────────────────────────────────
FEATURE_DISPLAY = {
    "Gender": "Gender", "Marital_Status": "Marital Status",
    "Dependents": "Dependents", "Education_Level": "Education Level",
    "Employment_Status": "Employment Status", "Employer_Category": "Employer Category",
    "Property_Area": "Property Area", "Loan_Purpose": "Loan Purpose",
    "Applicant_Income": "Applicant Income", "Coapplicant_Income": "Co-applicant Income",
    "Age": "Age", "Credit_Score": "Credit Score", "Existing_Loans": "Existing Loans",
    "DTI_Ratio": "Debt-to-Income Ratio", "Savings": "Savings",
    "Collateral_Value": "Collateral Value", "Loan_Amount": "Loan Amount",
    "Loan_Term": "Loan Term (months)", "Total_Income": "Total Income",
    "EMI": "Monthly EMI", "Balance_Income": "Balance Income",
    "Income_to_Loan_Ratio": "Income-to-Loan Ratio",
    "Savings_to_Loan_Ratio": "Savings-to-Loan Ratio",
    "Collateral_to_Loan_Ratio": "Collateral-to-Loan Ratio",
    "LoanAmount_log": "Loan Amount (log)", "Total_Income_log": "Total Income (log)",
    "Credit_Score_norm": "Credit Score (norm)",
}


# ── FPDF subclass with header / footer ────────────────────────────────────────

class _LoanPDF(FPDF):
    def header(self):
        # Bank icon + title
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 58, 95)
        self.cell(0, 10, "CreditWise AI - Loan Assessment Report",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5,
                  f"Generated {date.today().strftime('%B %d, %Y')}  |  "
                  "Powered by XGBoost + SHAP + Claude AI",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)
        self.set_draw_color(30, 58, 95)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(160, 160, 160)
        self.cell(
            0, 8,
            "DISCLAIMER: This AI-generated report is for informational purposes only "
            "and does not constitute a formal financial decision.",
            align="C",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 58, 95)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def two_col_row(self, label: str, value: str):
        # Strip any non-latin-1 chars for Helvetica compatibility
        value = value.encode("latin-1", errors="replace").decode("latin-1")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(65, 6, label + ":", new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.cell(55, 6, str(value), new_x="LMARGIN", new_y="NEXT")


# ── SHAP chart helper ─────────────────────────────────────────────────────────

def _shap_chart_png(shap_values: dict, top_n: int = 10) -> bytes:
    """Render a SHAP horizontal bar chart and return PNG bytes."""
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels = [FEATURE_DISPLAY.get(k, k.replace("_", " ")) for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = ["#10b981" if v > 0 else "#ef4444" for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, max(3, top_n * 0.38)))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.axvline(0, color="#374151", linewidth=0.8, linestyle="--", alpha=0.7)

    for bar, v in zip(bars, vals[::-1]):
        ax.text(
            v + (0.001 if v >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{'+' if v > 0 else ''}{v:.3f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=7.5, color="#374151",
        )

    ax.set_xlabel("SHAP Value  (← hurts approval | helps approval →)", fontsize=8)
    ax.set_title("Feature Contributions to This Prediction", fontsize=10, fontweight="bold", pad=8)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf_report(payload: dict, result: dict, explanation: dict) -> bytes:
    """
    Generate a complete PDF loan assessment report.

    Args:
        payload:     Flat dict of loan application inputs.
        result:      Prediction response dict (approved, probability, shap_values, ...).
        explanation: LLM explanation dict (explanation, key_factor, improvement_tips).

    Returns:
        PDF file as raw bytes -- pass directly to st.download_button.
    """
    def _s(t) -> str:
        """Sanitize to latin-1 safe string for Helvetica."""
        return str(t).encode("latin-1", errors="replace").decode("latin-1")

    approved    = result.get("approved", False)
    prob        = result.get("probability", 0.5)
    conf        = _s(result.get("confidence", "N/A"))
    shap_values = result.get("shap_values", {})

    pdf = _LoanPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Decision banner ───────────────────────────────────────────────────────
    if approved:
        pdf.set_fill_color(6, 95, 70)
        banner = f"  LOAN APPROVED  |  {prob:.1%} Probability  |  Confidence: {conf}  "
    else:
        pdf.set_fill_color(127, 29, 29)
        banner = f"  LOAN REJECTED  |  {prob:.1%} Probability  |  Confidence: {conf}  "

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 13, _s(banner), new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.ln(6)

    # ── Applicant Summary ─────────────────────────────────────────────────────
    pdf.section_title("Applicant Summary")
    rows = [
        ("Credit Score",        str(payload.get("credit_score", "-"))),
        ("Monthly Income",      f"Rs. {payload.get('applicant_income', 0):,.0f}"),
        ("Co-applicant Income", f"Rs. {payload.get('coapplicant_income', 0):,.0f}"),
        ("Loan Amount",         f"Rs. {payload.get('loan_amount', 0):,.0f}"),
        ("Loan Term",           f"{payload.get('loan_term', 0):.0f} months"),
        ("DTI Ratio",           f"{payload.get('dti_ratio', 0):.1%}"),
        ("Savings",             f"Rs. {payload.get('savings', 0):,.0f}"),
        ("Collateral Value",    f"Rs. {payload.get('collateral_value', 0):,.0f}"),
        ("Employment",          f"{payload.get('employment_status','-')} @ {payload.get('employer_category','-')}"),
        ("Property Area",       str(payload.get("property_area", "-"))),
        ("Loan Purpose",        str(payload.get("loan_purpose", "-"))),
        ("Education",           str(payload.get("education_level", "-"))),
    ]
    for lbl, val in rows:
        pdf.two_col_row(lbl, val)
    pdf.ln(4)

    # ── SHAP Chart ────────────────────────────────────────────────────────────
    if shap_values:
        pdf.section_title("SHAP Feature Contributions")
        chart_png = _shap_chart_png(shap_values)
        chart_buf = io.BytesIO(chart_png)
        pdf.image(chart_buf, x=10, w=190)
        pdf.ln(4)

    # ── AI Explanation ────────────────────────────────────────────────────────
    expl_text  = _s(explanation.get("explanation", "No explanation available."))
    key_factor = _s(explanation.get("key_factor", "-"))
    tips       = [_s(t) for t in explanation.get("improvement_tips", [])]

    pdf.section_title("AI-Powered Explanation (Claude Sonnet)")
    pdf.set_left_margin(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(10)
    pdf.multi_cell(190, 6, expl_text)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 95)
    pdf.set_x(10)
    pdf.cell(190, 6, "Key Factor:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(10)
    pdf.multi_cell(190, 6, key_factor)
    pdf.ln(3)

    # ── Improvement Tips ──────────────────────────────────────────────────────
    if tips:
        pdf.section_title("Actionable Recommendations")
        for i, tip in enumerate(tips, 1):
            tip_safe = tip.encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(10)
            pdf.multi_cell(190, 6, f"{i}. {tip_safe}")
        pdf.ln(2)

    # ── Model Info ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(
        0, 5,
        f"Model: XGBoost v{result.get('model_version', '1.0.0')}  |  "
        "Explainability: SHAP TreeExplainer  |  Language: Claude AI",
        new_x="LMARGIN", new_y="NEXT",
    )

    return bytes(pdf.output())
