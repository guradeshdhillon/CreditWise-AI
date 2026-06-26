"""
pages/2_🔄_WhatIf.py
CreditWise AI — What-If Scenario Analyzer

Shows which single feature changes would flip a REJECTED loan to APPROVED.
Displays probability curves for 6 key features with interactive Plotly charts.
Works in standalone mode (no FastAPI required).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import standalone_predict, FEATURE_DISPLAY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="What-If Analyzer | CreditWise AI",
    page_icon="🔄",
    layout="wide",
)

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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

.stApp {{ background: {_BG}; }}

[data-testid="stSidebar"] {{
    background: {_SURFACE} !important;
    border-right: 1px solid {_BORDER} !important;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{ color: {_TEXT2}; }}
[data-testid="stSidebarContent"] {{ padding: 16px 16px 0; }}

.flip-card {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-left: 4px solid {_GREEN};
    border-radius: 8px;
    padding: 16px 20px; margin: 8px 0;
}}
.flip-card h4 {{ color: {_TEXT1}; margin: 0 0 6px; font-size: 0.95rem; font-weight: 600; }}
.flip-card p  {{ color: {_TEXT2}; margin: 0; font-size: 0.88rem; }}
.flip-badge   {{ background: {_GREEN_BG}; color: {_GREEN}; border: 1px solid {_GREEN_D}; border-radius: 6px;
                padding: 3px 10px; font-size: 0.78rem; font-weight: 600; float: right; font-family: 'JetBrains Mono', monospace; }}

.no-flip-card {{
    background: {_SURFACE}; border: 1px solid {_BORDER}; border-left: 4px solid {_AMBER};
    border-radius: 8px; padding: 16px 20px; margin: 8px 0; color: {_TEXT2};
    font-size: 0.88rem;
}}

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

[data-testid="stExpander"] {{
    background: {_SURFACE} !important; border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{ color: {_TEXT2} !important; font-size: 0.88rem !important; }}

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
[data-testid="stSlider"] > div > div {{ background: {_RAISED} !important; }}

hr {{ border-color: {_BORDER2} !important; margin: 20px 0 !important; }}

.stDataFrame {{ border: 1px solid {_BORDER} !important; border-radius: 8px !important; }}
.stAlert {{ border-radius: 6px !important; font-size: 0.87rem !important; }}
.stProgress > div > div {{ background: {_BLUE} !important; border-radius: 4px; }}
[data-testid="stProgressBar"] {{ background: {_RAISED} !important; border-radius: 4px; }}
.stSpinner {{ color: {_BLUE} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:20px 0 16px; border-bottom: 1px solid {_BORDER2}; margin-bottom: 24px;'>
  <h1 style='font-size:1.6rem; font-weight:700; color:{_TEXT1}; margin:0; display:flex; align-items:center; gap:10px;'>
    <span style='background:{_SURFACE}; border:1px solid {_BORDER}; padding:4px 8px; border-radius:6px; font-size:1.1rem;'>🔄</span>
    What-If Scenario Analyzer
  </h1>
  <p style='color:{_TEXT3}; font-size:0.85rem; margin-top:6px;'>
    Discover which single change would flip a <span style='color:{_RED}; font-weight:600;'>Rejected</span>
    application to <span style='color:{_GREEN}; font-weight:600;'>Approved</span>.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sweepable feature definitions ─────────────────────────────────────────────
SWEEPABLE = {
    "credit_score": {
        "label": "Credit Score",
        "vals": list(range(300, 901, 15)),
        "fmt": lambda v: f"{v:.0f}",
        "hint": "Higher is better. 750+ is typically strong.",
    },
    "dti_ratio": {
        "label": "Debt-to-Income Ratio",
        "vals": [round(v, 2) for v in np.arange(0.05, 0.96, 0.04)],
        "fmt": lambda v: f"{v:.0%}",
        "hint": "Lower is better. Aim for < 40%.",
    },
    "savings": {
        "label": "Savings Amount (₹)",
        "vals": list(range(0, 200_001, 5_000)),
        "fmt": lambda v: f"₹{v:,.0f}",
        "hint": "Higher savings reduce lender risk.",
    },
    "loan_amount": {
        "label": "Loan Amount (₹)",
        "vals": list(range(5_000, 300_001, 5_000)),
        "fmt": lambda v: f"₹{v:,.0f}",
        "hint": "Lower loan amounts are easier to approve.",
    },
    "coapplicant_income": {
        "label": "Co-applicant Income (₹)",
        "vals": list(range(0, 30_001, 1_000)),
        "fmt": lambda v: f"₹{v:,.0f}",
        "hint": "Adding a co-applicant income strengthens the application.",
    },
    "existing_loans": {
        "label": "Existing Active Loans",
        "vals": list(range(0, 11, 1)),
        "fmt": lambda v: f"{v:.0f}",
        "hint": "Fewer existing loans improve approval chances.",
    },
}

# ── Load payload from session state or form ───────────────────────────────────
st.markdown(f"<div style='color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; border-bottom:1px solid {_BORDER2}; padding-bottom:8px;'>Application Under Analysis</div>", unsafe_allow_html=True)

has_session = "last_payload" in st.session_state and "last_result" in st.session_state

if has_session:
    payload = st.session_state["last_payload"].copy()
    base_result = st.session_state["last_result"]
    base_prob = base_result.get("probability", 0.5)
    base_approved = base_result.get("approved", True)

    status_color = _GREEN if base_approved else _RED
    status_text  = "APPROVED" if base_approved else "REJECTED"
    bg_color = _GREEN_BG if base_approved else _RED_BG
    border_color = _GREEN_D if base_approved else _RED_D
    
    st.markdown(
        f"<div style='background:{bg_color};border:1px solid {border_color};"
        f"border-radius:8px;padding:14px 20px;'>"
        f"<b style='color:{_TEXT1};'>Using application from main page</b> &nbsp;·&nbsp; "
        f"Current status: <b style='color:{status_color}; font-family:\"JetBrains Mono\",monospace;'>{status_text}</b> "
        f"<span style='color:{_TEXT3};'>({base_prob:.1%} probability)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='color:{_TEXT3}; font-size:0.8rem; margin-top:8px;'>Or enter a different application below and click <b>Run Analysis</b>.</p>", unsafe_allow_html=True)
else:
    st.info("💡 No previous application found. Fill in the form below and click **Run Analysis**.")
    payload = {
        "gender": "Male", "marital_status": "Married", "dependents": 1,
        "education_level": "Graduate", "employment_status": "Salaried",
        "employer_category": "Private", "age": 30,
        "applicant_income": 10000.0, "coapplicant_income": 0.0,
        "loan_amount": 50000.0, "loan_term": 60.0, "existing_loans": 2,
        "dti_ratio": 0.55, "savings": 5000.0, "collateral_value": 0.0,
        "credit_score": 620, "property_area": "Rural", "loan_purpose": "Personal",
    }
    base_prob = None

with st.expander("Edit Application", expanded=not has_session):
    ec1, ec2, ec3 = st.columns(3)
    payload["gender"]            = ec1.selectbox("Gender", ["Male","Female"],
                                    index=["Male","Female"].index(payload.get("gender","Male")))
    payload["marital_status"]    = ec2.selectbox("Marital Status", ["Married","Single"],
                                    index=["Married","Single"].index(payload.get("marital_status","Married")))
    payload["education_level"]   = ec3.selectbox("Education", ["Graduate","Not Graduate"],
                                    index=["Graduate","Not Graduate"].index(payload.get("education_level","Graduate")))

    ec4, ec5, ec6 = st.columns(3)
    payload["employment_status"] = ec4.selectbox("Employment", ["Salaried","Self-employed","Contract","Unemployed"],
                                    index=["Salaried","Self-employed","Contract","Unemployed"].index(payload.get("employment_status","Salaried")))
    payload["property_area"]     = ec5.selectbox("Property Area", ["Urban","Semiurban","Rural"],
                                    index=["Urban","Semiurban","Rural"].index(payload.get("property_area","Urban")))
    payload["loan_purpose"]      = ec6.selectbox("Loan Purpose", ["Home","Car","Personal","Education","Business"],
                                    index=["Home","Car","Personal","Education","Business"].index(payload.get("loan_purpose","Personal")))

    en1, en2, en3, en4 = st.columns(4)
    payload["applicant_income"]   = float(en1.number_input("Income (₹)", 1000, 500000, int(payload.get("applicant_income",10000)), 500))
    payload["credit_score"]       = int(en2.slider("Credit Score", 300, 900, int(payload.get("credit_score",620)), 10))
    payload["loan_amount"]        = float(en3.number_input("Loan Amount (₹)", 5000, 500000, int(payload.get("loan_amount",50000)), 1000))
    payload["dti_ratio"]          = float(en4.slider("DTI Ratio", 0.05, 0.95, float(payload.get("dti_ratio",0.55)), 0.01))

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("Run What-If Analysis", use_container_width=True)

# ── Chart theme ───────────────────────────────────────────────────────────────
_CHART_BASE = dict(
    plot_bgcolor=_SURFACE,
    paper_bgcolor=_BG,
    font=dict(color=_TEXT2, family="Inter, sans-serif", size=11),
    xaxis=dict(gridcolor=_BORDER2, zerolinecolor=_BORDER, tickfont=dict(color=_TEXT3)),
    yaxis=dict(gridcolor=_BORDER2, tickfont=dict(color=_TEXT2)),
    hoverlabel=dict(bgcolor=_SURFACE2, font_color=_TEXT1, bordercolor=_BORDER),
    margin=dict(l=10, r=20, t=30, b=50),
)

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn or has_session:
    # Get base prediction
    with st.spinner("Computing base prediction…"):
        base_result_fresh = standalone_predict(payload)
    base_prob     = base_result_fresh["probability"]
    base_approved = base_result_fresh["approved"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; border-bottom:1px solid {_BORDER2}; padding-bottom:8px;'>Feature Sweep Results</div>", unsafe_allow_html=True)
    st.caption(
        f"Base probability: <b style='color:{_TEXT1}; font-family:\"JetBrains Mono\",monospace;'>{base_prob:.1%}</b> "
        f"({'<span style=\"color:'+_GREEN+';\">APPROVED</span>' if base_approved else '<span style=\"color:'+_RED+';\">REJECTED</span>'}). "
        "The charts below show how each feature independently affects the approval probability."
    )

    # ── Sweep all features ────────────────────────────────────────────────────
    all_curves: dict[str, dict] = {}
    flip_results: list[dict] = []

    progress_bar = st.progress(0, text="Sweeping features…")
    n_features = len(SWEEPABLE)

    for fi, (feat_key, feat_cfg) in enumerate(SWEEPABLE.items()):
        vals = feat_cfg["vals"]
        probs = []
        flip_val = None
        flip_prob = None

        for v in vals:
            test_payload = {**payload, feat_key: v}
            res = standalone_predict(test_payload)
            probs.append(res["probability"])
            if not base_approved and res["approved"] and flip_val is None:
                flip_val  = v
                flip_prob = res["probability"]

        all_curves[feat_key] = {"vals": vals, "probs": probs, "cfg": feat_cfg}

        if flip_val is not None:
            current_val = payload.get(feat_key, vals[0])
            change = flip_val - current_val
            flip_results.append({
                "feature":     feat_key,
                "label":       feat_cfg["label"],
                "current_val": current_val,
                "flip_val":    flip_val,
                "flip_prob":   flip_prob,
                "change":      abs(change),
                "fmt":         feat_cfg["fmt"],
                "hint":        feat_cfg["hint"],
            })
        progress_bar.progress((fi + 1) / n_features, text=f"Analysed: {feat_cfg['label']}")

    progress_bar.empty()

    # ── Flip summary ──────────────────────────────────────────────────────────
    st.markdown(f"<div style='margin-top: 10px;'><b style='color:{_TEXT1}; font-size:1rem;'>Minimum Changes to Get Approved</b></div>", unsafe_allow_html=True)

    if not base_approved and flip_results:
        flip_results.sort(key=lambda x: x["change"])
        st.success(f"Found **{len(flip_results)}** feature change(s) that would flip this application to APPROVED!")

        cols = st.columns(min(len(flip_results), 3))
        for i, fr in enumerate(flip_results[:3]):
            with cols[i]:
                current_fmt = fr["fmt"](fr["current_val"])
                flip_fmt    = fr["fmt"](fr["flip_val"])
                st.markdown(
                    f'<div class="flip-card">'
                    f'<span class="flip-badge">FLIP #{i+1}</span>'
                    f'<h4>{fr["label"]}</h4>'
                    f'<p>Change: <b style="font-family:\'JetBrains Mono\';">{current_fmt}</b> → <b style="color:{_GREEN}; font-family:\'JetBrains Mono\';">{flip_fmt}</b></p>'
                    f'<p style="margin-top:6px;color:{_TEXT3};font-size:0.75rem;">{fr["hint"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if len(flip_results) > 3:
            with st.expander(f"See all {len(flip_results)} flip options"):
                rows = []
                for fr in flip_results:
                    rows.append({
                        "Feature":       fr["label"],
                        "Current Value": fr["fmt"](fr["current_val"]),
                        "Required Value": fr["fmt"](fr["flip_val"]),
                        "New Probability": f"{fr['flip_prob']:.1%}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    elif base_approved:
        st.success("This application is already APPROVED! Charts show sensitivity analysis.")
    else:
        st.warning("No single feature change is sufficient to flip the decision. Multiple improvements may be needed.")
        st.markdown(f'<div class="no-flip-card">Consider improving multiple factors simultaneously: credit score, DTI ratio, and savings together often make the difference.</div>', unsafe_allow_html=True)

    # ── Probability curves ────────────────────────────────────────────────────
    st.markdown(f"<div style='margin-top: 30px;'><b style='color:{_TEXT1}; font-size:1rem;'>Probability Curves — How Each Feature Affects Approval</b></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_keys = list(SWEEPABLE.keys())
    tabs = st.tabs([SWEEPABLE[k]["label"] for k in tab_keys])

    for tab, feat_key in zip(tabs, tab_keys):
        with tab:
            curve = all_curves[feat_key]
            vals_disp = [curve["cfg"]["fmt"](v) for v in curve["vals"]]

            fig = go.Figure()

            # Threshold line
            fig.add_hline(y=0.5, line_dash="dash", line_color=_BORDER,
                          annotation_text="Decision Threshold (50%)",
                          annotation_font_color=_TEXT3, annotation_position="top right")

            # Probability curve
            fig.add_trace(go.Scatter(
                x=curve["vals"], y=curve["probs"],
                mode="lines+markers",
                line=dict(color=_BLUE, width=2.5),
                marker=dict(size=4, color=_BLUE),
                name="Approval Probability",
                hovertemplate=f"{curve['cfg']['label']}: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>",
            ))

            # Current value marker
            cur_val = payload.get(feat_key)
            if cur_val is not None and cur_val in curve["vals"]:
                idx = curve["vals"].index(cur_val)
                fig.add_trace(go.Scatter(
                    x=[cur_val], y=[curve["probs"][idx]],
                    mode="markers",
                    marker=dict(size=10, color=_AMBER, symbol="diamond",
                                line=dict(color=_BG, width=1)),
                    name=f"Current ({curve['cfg']['fmt'](cur_val)})",
                ))

            # Flip marker
            fr_match = next((fr for fr in flip_results if fr["feature"] == feat_key), None)
            if fr_match:
                fig.add_trace(go.Scatter(
                    x=[fr_match["flip_val"]], y=[fr_match["flip_prob"]],
                    mode="markers",
                    marker=dict(size=12, color=_GREEN, symbol="star",
                                line=dict(color=_BG, width=1)),
                    name=f"Flip Point ({curve['cfg']['fmt'](fr_match['flip_val'])})",
                ))

            fig.update_layout(
                **_CHART_BASE,
                xaxis=dict(title=dict(text=curve["cfg"]["label"], font=dict(color=_TEXT3, size=10)),
                           gridcolor=_BORDER2, tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
                yaxis=dict(title=dict(text="Approval Probability", font=dict(color=_TEXT3, size=10)), range=[-0.03, 1.05],
                           tickformat=".0%", gridcolor=_BORDER2,
                           tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(color=_TEXT2, size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"💡 {curve['cfg']['hint']}")
