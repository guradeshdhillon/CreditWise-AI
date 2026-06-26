"""
streamlit_app.py  —  CreditWise AI
Professional dark-theme loan intelligence dashboard.
Design: GitHub Dark palette · Inter typography · clean card layout.
"""

import sys
import os
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import standalone_predict, standalone_explain, FEATURE_DISPLAY, load_artifacts
from pdf_report import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise AI | Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000/api/v1"

# ── Design tokens ─────────────────────────────────────────────────────────────
_BG       = "#0d1117"
_SURFACE  = "#161b22"
_SURFACE2 = "#1c2128"
_RAISED   = "#21262d"
_BORDER   = "#30363d"
_BORDER2  = "#21262d"
_TEXT1    = "#e6edf3"
_TEXT2    = "#8b949e"
_TEXT3    = "#6e7681"
_BLUE     = "#1f6feb"
_BLUE_L   = "#388bfd"
_GREEN    = "#3fb950"
_GREEN_D  = "#2ea043"
_GREEN_BG = "#0f2d1a"
_RED      = "#f85149"
_RED_D    = "#da3633"
_RED_BG   = "#2d0f0f"
_AMBER    = "#e3b341"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* ── Base ──────────────────────────────────────────────── */
.stApp {{ background: {_BG}; }}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {_SURFACE} !important;
    border-right: 1px solid {_BORDER} !important;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{ color: {_TEXT2}; }}
[data-testid="stSidebarContent"] {{ padding: 16px 16px 0; }}

/* ── Metrics ────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{ color: {_TEXT1} !important; font-weight: 700 !important; font-size: 1.4rem !important; }}
[data-testid="stMetricLabel"] {{ color: {_TEXT3} !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
[data-testid="stMetricDelta"] {{ font-size: 0.75rem !important; }}

/* ── Inputs ─────────────────────────────────────────────── */
.stSelectbox label, .stNumberInput label, .stSlider label {{
    color: {_TEXT3} !important; font-size: 0.75rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
.stSelectbox > div > div {{
    background: {_RAISED} !important; border-color: {_BORDER} !important;
    color: {_TEXT1} !important; border-radius: 6px !important;
}}
.stNumberInput > div > div > input {{
    background: {_RAISED} !important; border-color: {_BORDER} !important;
    color: {_TEXT1} !important; border-radius: 6px !important;
}}

/* ── Slider ─────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div {{ background: {_RAISED} !important; }}

/* ── Dividers ───────────────────────────────────────────── */
hr {{ border-color: {_BORDER2} !important; margin: 20px 0 !important; }}

/* ── Expander ───────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {_SURFACE} !important; border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{ color: {_TEXT2} !important; font-size: 0.88rem !important; }}

/* ── Alerts ─────────────────────────────────────────────── */
.stAlert {{ border-radius: 6px !important; font-size: 0.87rem !important; }}

/* ── Progress ───────────────────────────────────────────── */
.stProgress > div > div {{ background: {_BLUE} !important; border-radius: 4px; }}
[data-testid="stProgressBar"] {{ background: {_RAISED} !important; border-radius: 4px; }}

/* ── Spinner ─────────────────────────────────────────────── */
.stSpinner {{ color: {_BLUE} !important; }}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {{
    background: {_BLUE} !important; color: #ffffff !important;
    border: none !important; border-radius: 6px !important;
    padding: 10px 20px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; width: 100%; letter-spacing: 0.01em;
    transition: background 0.15s ease;
    font-family: 'Inter', sans-serif !important;
}}
.stButton > button:hover {{ background: {_BLUE_L} !important; }}
.stButton > button:active {{ background: {_BLUE} !important; transform: none !important; }}

.stDownloadButton > button {{
    background: transparent !important; color: {_GREEN} !important;
    border: 1px solid {_GREEN_D} !important; border-radius: 6px !important;
    font-weight: 500 !important; font-size: 0.87rem !important;
    transition: background 0.15s ease; padding: 8px 18px !important;
    font-family: 'Inter', sans-serif !important;
}}
.stDownloadButton > button:hover {{ background: {_GREEN_BG} !important; }}

/* ── Dataframe ──────────────────────────────────────────── */
.stDataFrame {{ border: 1px solid {_BORDER} !important; border-radius: 8px !important; }}

/* ── Custom components ──────────────────────────────────── */

.page-header {{
    display: flex; align-items: center; gap: 14px;
    padding: 24px 0 20px; border-bottom: 1px solid {_BORDER2}; margin-bottom: 24px;
}}
.logo-mark {{
    width: 42px; height: 42px; background: {_BLUE};
    border-radius: 9px; display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 800; color: white; flex-shrink: 0;
    letter-spacing: -0.06em; font-family: 'JetBrains Mono', monospace;
}}
.page-title {{
    margin: 0; font-size: 1.45rem; font-weight: 700;
    color: {_TEXT1}; letter-spacing: -0.02em;
}}
.page-sub {{ margin: 2px 0 0; color: {_TEXT3}; font-size: 0.82rem; font-weight: 400; }}
.version-badge {{
    margin-left: auto; background: {_RAISED}; color: {_TEXT3};
    border: 1px solid {_BORDER}; border-radius: 20px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 500; font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}}

.section-label {{
    color: {_TEXT3}; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding-bottom: 8px; border-bottom: 1px solid {_BORDER2};
    margin: 24px 0 14px;
}}

.card {{
    background: {_SURFACE}; border: 1px solid {_BORDER};
    border-radius: 8px; padding: 18px 22px; margin: 10px 0;
}}

.decision-approved {{
    background: {_GREEN_BG}; border: 1px solid {_GREEN_D};
    border-left: 4px solid {_GREEN}; border-radius: 8px;
    padding: 20px 26px; margin: 18px 0;
}}
.decision-rejected {{
    background: {_RED_BG}; border: 1px solid {_RED_D};
    border-left: 4px solid {_RED}; border-radius: 8px;
    padding: 20px 26px; margin: 18px 0;
}}
.decision-eyebrow {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; margin: 0 0 5px; opacity: 0.7;
}}
.decision-title {{
    font-size: 1.35rem; font-weight: 700; color: {_TEXT1};
    letter-spacing: -0.02em; margin: 0 0 4px;
}}
.decision-meta {{
    font-size: 0.83rem; color: {_TEXT2}; margin: 0;
    font-family: 'JetBrains Mono', monospace;
}}

.shap-table {{
    background: {_SURFACE}; border: 1px solid {_BORDER};
    border-radius: 8px; overflow: hidden; margin: 8px 0;
}}
.shap-thead {{
    display: flex; justify-content: space-between;
    padding: 7px 16px; background: {_RAISED};
    border-bottom: 1px solid {_BORDER};
}}
.shap-th {{ color: {_TEXT3}; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; }}
.shap-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; border-bottom: 1px solid {_BORDER2};
}}
.shap-row:last-child {{ border-bottom: none; }}
.shap-feature {{ color: {_TEXT1}; font-size: 0.85rem; }}
.shap-pos {{ color: {_GREEN}; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }}
.shap-neg {{ color: {_RED}; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }}

.explanation-text {{
    color: #c9d1d9; font-size: 0.9rem; line-height: 1.75;
}}
.kf-label {{
    color: {_TEXT3}; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}}
.kf-value {{ color: {_TEXT1}; font-size: 0.88rem; }}

.tip-item {{
    display: flex; gap: 12px; align-items: flex-start;
    padding: 9px 0; border-bottom: 1px solid {_BORDER2};
}}
.tip-item:last-child {{ border-bottom: none; }}
.tip-num {{
    color: {_BLUE_L}; font-weight: 700; font-size: 0.82rem;
    flex-shrink: 0; font-family: 'JetBrains Mono', monospace;
    min-width: 22px;
}}
.tip-text {{ color: {_TEXT2}; font-size: 0.87rem; line-height: 1.6; }}

.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; }}
.dot-green {{ background: {_GREEN}; }}
.dot-amber {{ background: {_AMBER}; }}

.stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid {_BORDER2};
    font-size: 0.82rem;
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-key {{ color: {_TEXT3}; }}
.stat-val {{ color: {_TEXT1}; font-family: 'JetBrains Mono', monospace; font-weight: 500; }}

.nav-section {{ margin-top: 20px; padding-top: 16px; border-top: 1px solid {_BORDER2}; }}
.nav-item {{
    display: block; padding: 7px 10px; margin: 3px 0;
    color: {_TEXT2} !important; font-size: 0.85rem;
    border-radius: 6px; text-decoration: none;
    transition: background 0.1s;
}}
.nav-item:hover {{ background: {_RAISED}; color: {_TEXT1} !important; }}

.notice-box {{
    background: {_SURFACE}; border: 1px solid {_BORDER};
    border-radius: 6px; padding: 10px 14px; margin: 8px 0;
    color: {_TEXT2}; font-size: 0.83rem; line-height: 1.5;
}}

.stPlotlyChart {{ border-radius: 8px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Chart theme ───────────────────────────────────────────────────────────────
_CHART_BASE = dict(
    plot_bgcolor=_SURFACE,
    paper_bgcolor=_SURFACE,
    font=dict(color=_TEXT2, family="Inter, sans-serif", size=11),
    xaxis=dict(gridcolor=_BORDER2, zerolinecolor=_BORDER, tickfont=dict(color=_TEXT3)),
    yaxis=dict(gridcolor=_BORDER2, tickfont=dict(color=_TEXT2)),
    hoverlabel=dict(bgcolor=_SURFACE2, font_color=_TEXT1, bordercolor=_BORDER),
    margin=dict(l=10, r=70, t=40, b=50),
)


# ── SHAP waterfall chart ──────────────────────────────────────────────────────
def shap_waterfall_chart(shap_dict: dict) -> go.Figure:
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    features = [FEATURE_DISPLAY.get(k, k.replace("_", " ")) for k, _ in sorted_items]
    values   = [v for _, v in sorted_items]
    colors   = [_GREEN if v > 0 else _RED for v in values]
    texts    = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=features[::-1],
        orientation="h",
        marker_color=colors[::-1],
        marker_line_width=0,
        text=texts[::-1],
        textposition="outside",
        textfont=dict(color=_TEXT2, size=10, family="JetBrains Mono"),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=_BORDER, line_width=1)
    fig.update_layout(
        **_CHART_BASE,
        title=dict(
            text="Feature Contributions (SHAP)",
            font=dict(color=_TEXT1, size=13, family="Inter"),
            x=0, pad=dict(l=0),
        ),
        xaxis=dict(
            title=dict(text="← Reduces Approval  ·  Increases Approval →", font=dict(color=_TEXT3, size=10)),
            gridcolor=_BORDER2, zerolinecolor=_BORDER,
            tickfont=dict(color=_TEXT3, family="JetBrains Mono"),
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_TEXT1, size=11)),
        height=400,
        margin=dict(l=10, r=80, t=46, b=50),
    )
    return fig


# ── Prediction helpers ────────────────────────────────────────────────────────
def run_predict(payload: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return standalone_predict(payload)


def run_explain(payload: dict, result: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}/explain",
                          json={"application": payload, "prediction": result}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return standalone_explain(payload, result)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="padding: 8px 0 16px; border-bottom: 1px solid {_BORDER2}; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="logo-mark">CW</div>
            <div>
                <div style="color: {_TEXT1}; font-weight: 700; font-size: 0.95rem;">CreditWise AI</div>
                <div style="color: {_TEXT3}; font-size: 0.72rem; margin-top: 1px;">Loan Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API status + model info
    api_live = False
    perf = {}
    model_ver = "1.0.0"
    try:
        h = requests.get(f"{API_URL}/health", timeout=3)
        if h.status_code == 200:
            api_live = True
            data = h.json()
            perf = data.get("performance", {})
            model_ver = data.get("model_version", "1.0.0")
    except Exception:
        pass

    if not api_live:
        try:
            _, _, info, _ = load_artifacts()
            perf = info.get("performance", {})
            model_ver = info.get("model_version", "1.0.0")
        except Exception:
            pass

    dot_cls = "dot-green" if api_live else "dot-amber"
    mode_label = "API Live" if api_live else "Standalone"

    st.markdown(f"""
    <div style="margin-bottom: 4px;">
        <div style="color: {_TEXT3}; font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-bottom: 10px;">Model Status</div>
        <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid {_BORDER2};">
            <span class="dot {dot_cls}"></span>
            <span style="color: {_TEXT1}; font-size: 0.85rem; font-weight: 500;">{mode_label}</span>
        </div>
        <div class="stat-row"><span class="stat-key">Algorithm</span><span class="stat-val">XGBoost</span></div>
        <div class="stat-row"><span class="stat-key">Version</span><span class="stat-val">v{model_ver}</span></div>
        <div class="stat-row"><span class="stat-key">Explainability</span><span class="stat-val">SHAP</span></div>
        <div class="stat-row"><span class="stat-key">Language Model</span><span class="stat-val">Claude AI</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top: 18px; margin-bottom: 4px;">
        <div style="color: {_TEXT3}; font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-bottom: 10px;">Performance</div>
        <div class="stat-row"><span class="stat-key">ROC-AUC</span>
            <span class="stat-val" style="color:{_GREEN};">{perf.get('roc_auc', 0):.4f}</span></div>
        <div class="stat-row"><span class="stat-key">F1 Score</span>
            <span class="stat-val">{perf.get('f1', 0):.4f}</span></div>
        <div class="stat-row"><span class="stat-key">Accuracy</span>
            <span class="stat-val">{perf.get('accuracy', 0):.4f}</span></div>
        <div class="stat-row"><span class="stat-key">Precision</span>
            <span class="stat-val">{perf.get('precision', 0):.4f}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="nav-section">
        <div style="color: {_TEXT3}; font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-bottom: 8px;">Pages</div>
        <a class="nav-item" href="/What_If_Analyzer" target="_self">↔ What-If Analyzer</a>
        <a class="nav-item" href="/Fairness_Audit" target="_self">⊖ Fairness Audit</a>
        {"<a class='nav-item' href='http://localhost:8000/docs' target='_blank'>⊕ API Docs</a>" if api_live else ""}
    </div>
    <div style="margin-top: 20px; padding-top: 14px; border-top: 1px solid {_BORDER2};
                color: {_TEXT3}; font-size: 0.72rem; line-height: 1.6;">
        Built by <span style="color:{_TEXT2}; font-weight: 500;">Guradesh Dhillon</span><br>
        Computer Engineering · AI & DS
    </div>
    """, unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="logo-mark">CW</div>
    <div>
        <div class="page-title">CreditWise AI</div>
        <div class="page-sub">Loan Approval Intelligence Platform · XGBoost + SHAP + Claude AI</div>
    </div>
    <span class="version-badge">v{model_ver}</span>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
with st.form("loan_form"):
    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    gender      = c1.selectbox("Gender", ["Male", "Female"])
    marital     = c2.selectbox("Marital Status", ["Married", "Single"])
    dependents  = c3.number_input("Dependents", 0, 10, 0, step=1)
    age         = c4.number_input("Age", 18, 80, 30, step=1)

    c5, c6 = st.columns(2)
    education   = c5.selectbox("Education Level", ["Graduate", "Not Graduate"])
    property_a  = c6.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown('<div class="section-label">Employment</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    employment  = c7.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    employer    = c8.selectbox("Employer Category", ["Private", "Government", "MNC", "NGO"])
    purpose     = c9.selectbox("Loan Purpose", ["Home", "Car", "Personal", "Education", "Business"])

    st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
    c10, c11 = st.columns(2)
    app_income   = c10.number_input("Monthly Income (INR)", 1000, 500_000, 15000, step=500)
    coapp_income = c11.number_input("Co-applicant Income (INR)", 0, 200_000, 0, step=500)

    c12, c13, c14 = st.columns(3)
    credit_score   = c12.slider("Credit Score", 300, 900, 700, step=10)
    existing_loans = c13.number_input("Existing Active Loans", 0, 20, 0, step=1)
    dti_ratio      = c14.slider("Debt-to-Income Ratio", 0.05, 0.95, 0.35, step=0.01)

    c15, c16 = st.columns(2)
    savings        = c15.number_input("Savings (INR)", 0, 10_000_000, 20000, step=1000)
    collateral_val = c16.number_input("Collateral Value (INR)", 0, 10_000_000, 0, step=1000)

    st.markdown('<div class="section-label">Loan Details</div>', unsafe_allow_html=True)
    c17, c18 = st.columns(2)
    loan_amount = c17.number_input("Loan Amount (INR)", 5000, 5_000_000, 25000, step=1000)
    loan_term   = c18.selectbox("Loan Term (months)", [12, 24, 36, 60, 84, 120, 180, 240, 360], index=3)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Assess Loan Application", use_container_width=True)


# ── Payload ───────────────────────────────────────────────────────────────────
payload = {
    "gender": gender, "marital_status": marital, "dependents": int(dependents),
    "education_level": education, "employment_status": employment,
    "employer_category": employer, "age": int(age),
    "applicant_income": float(app_income), "coapplicant_income": float(coapp_income),
    "loan_amount": float(loan_amount), "loan_term": float(loan_term),
    "existing_loans": int(existing_loans), "dti_ratio": float(dti_ratio),
    "savings": float(savings), "collateral_value": float(collateral_val),
    "credit_score": int(credit_score), "property_area": property_a, "loan_purpose": purpose,
}


# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    with st.spinner("Running prediction…"):
        result = run_predict(payload)

    if "error" in result:
        st.error(f"Prediction error: {result['error']}")
        st.stop()

    st.session_state["last_payload"] = payload
    st.session_state["last_result"]  = result

    approved  = result["approved"]
    prob      = result["probability"]
    conf      = result["confidence"]
    shap_dict = result.get("shap_values", {})
    top_facs  = result.get("top_factors", [])

    # ── Decision banner ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Decision</div>', unsafe_allow_html=True)
    if approved:
        st.markdown(f"""
        <div class="decision-approved">
            <div class="decision-eyebrow" style="color:{_GREEN};">Approved</div>
            <div class="decision-title">Loan Application Approved</div>
            <div class="decision-meta">
                Approval probability: {prob:.4f} &nbsp;·&nbsp; Confidence: {conf}
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="decision-rejected">
            <div class="decision-eyebrow" style="color:{_RED};">Rejected</div>
            <div class="decision-title">Loan Application Rejected</div>
            <div class="decision-meta">
                Approval probability: {prob:.4f} &nbsp;·&nbsp; Confidence: {conf}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    if top_facs or shap_dict:
        st.markdown('<div class="section-label">SHAP Contribution Analysis</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        pos_items = [f for f in top_facs if f.get("direction") == "positive"]
        neg_items = [f for f in top_facs if f.get("direction") == "negative"]

        with col_l:
            st.markdown(f"<div style='color:{_TEXT3}; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;'>Supporting Factors</div>", unsafe_allow_html=True)
            if pos_items:
                rows = "".join(
                    f'<div class="shap-row"><span class="shap-feature">{FEATURE_DISPLAY.get(f["feature"], f["feature"].replace("_"," "))}</span>'
                    f'<span class="shap-pos">+{f["value"]:.3f}</span></div>'
                    for f in pos_items
                )
                st.markdown(f'<div class="shap-table"><div class="shap-thead"><span class="shap-th">Feature</span><span class="shap-th">SHAP</span></div>{rows}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{_TEXT3}; font-size:0.84rem; padding:8px 0;">No significant positive factors.</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown(f"<div style='color:{_TEXT3}; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;'>Risk Factors</div>", unsafe_allow_html=True)
            if neg_items:
                rows = "".join(
                    f'<div class="shap-row"><span class="shap-feature">{FEATURE_DISPLAY.get(f["feature"], f["feature"].replace("_"," "))}</span>'
                    f'<span class="shap-neg">{f["value"]:.3f}</span></div>'
                    for f in neg_items
                )
                st.markdown(f'<div class="shap-table"><div class="shap-thead"><span class="shap-th">Feature</span><span class="shap-th">SHAP</span></div>{rows}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{_TEXT3}; font-size:0.84rem; padding:8px 0;">No significant risk factors.</div>', unsafe_allow_html=True)

        if shap_dict:
            st.markdown(f"<div style='color:{_TEXT3}; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin: 18px 0 10px;'>Waterfall Chart</div>", unsafe_allow_html=True)
            st.plotly_chart(shap_waterfall_chart(shap_dict), use_container_width=True)

    # ── LLM explanation ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">AI Explanation</div>', unsafe_allow_html=True)
    with st.spinner("Generating explanation…"):
        explanation = run_explain(payload, result)

    expl_text  = explanation.get("explanation", "—")
    key_factor = explanation.get("key_factor", "—")
    tips       = explanation.get("improvement_tips", [])

    st.markdown(f"""
    <div class="card">
        <div class="explanation-text">{expl_text}</div>
        {"" if not key_factor or key_factor == "—" else f'<div style="margin-top:14px; padding-top:14px; border-top:1px solid {_BORDER2};"><div class="kf-label">Key Factor</div><div class="kf-value">{key_factor}</div></div>'}
    </div>
    """, unsafe_allow_html=True)

    if tips:
        st.markdown(f"""
        <div style="color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin: 16px 0 8px;">
            Recommendations
        </div>
        <div class="card" style="padding: 14px 22px;">
        {"".join(f'<div class="tip-item"><span class="tip-num">0{i}.</span><span class="tip-text">{tip}</span></div>' for i, tip in enumerate(tips, 1))}
        </div>""", unsafe_allow_html=True)

    # ── PDF Download ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Report</div>', unsafe_allow_html=True)
    col_dl, col_note = st.columns([1, 2])
    with col_dl:
        try:
            pdf_bytes = generate_pdf_report(payload, result, explanation)
            st.download_button(
                label="Download PDF Assessment",
                data=pdf_bytes,
                file_name=f"creditwise_assessment_{'approved' if approved else 'rejected'}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF error: {e}")
    with col_note:
        if not approved:
            st.markdown(f'<div class="notice-box">Use the <b>What-If Analyzer</b> page to explore which changes would flip this decision to Approved.</div>', unsafe_allow_html=True)
