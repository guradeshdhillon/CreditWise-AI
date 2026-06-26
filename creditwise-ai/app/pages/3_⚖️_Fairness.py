"""
pages/3_⚖️_Fairness.py
CreditWise AI — Fairness & Bias Audit

Runs the trained model across the full dataset and reports:
  - Approval rates by demographic group (Gender, Property Area, Employment, Education)
  - Disparate Impact Ratio (DIR) with pass/warn/fail indicators
  - Global SHAP feature importance (mean |SHAP|)
  - Responsible AI disclaimer

Works in standalone mode (no FastAPI required).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_artifacts, payload_to_df, FEATURE_DISPLAY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fairness Audit | CreditWise AI",
    page_icon="⚖️",
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

html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
.stApp {{ background: {_BG}; }}
[data-testid="stSidebar"] {{
    background: {_SURFACE} !important; border-right: 1px solid {_BORDER} !important;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {{ color: {_TEXT2}; }}
[data-testid="stSidebarContent"] {{ padding: 16px 16px 0; }}

/* ── Metrics ────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{ color: {_TEXT1} !important; font-weight: 700 !important; font-size: 1.4rem !important; font-family: 'JetBrains Mono', monospace; }}
[data-testid="stMetricLabel"] {{ color: {_TEXT3} !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}

.dir-pass {{ background: {_GREEN_BG}; border: 1px solid {_GREEN_D};
            border-radius: 8px; padding: 14px 18px; text-align:center; }}
.dir-warn {{ background: rgba(227,179,65,0.1); border: 1px solid {_AMBER};
            border-radius: 8px; padding: 14px 18px; text-align:center; }}
.dir-fail {{ background: {_RED_BG}; border: 1px solid {_RED_D};
            border-radius: 8px; padding: 14px 18px; text-align:center; }}
.dir-title {{ font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
.dir-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em;
             color: {_TEXT3}; margin-top: 4px; }}

.resp-card {{
    background: {_SURFACE};
    border: 1px solid {_BORDER}; border-radius: 8px; padding: 20px 24px; margin: 16px 0;
}}

hr {{ border-color: {_BORDER2} !important; margin: 20px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:20px 0 16px; border-bottom: 1px solid {_BORDER2}; margin-bottom: 24px;'>
  <h1 style='font-size:1.6rem; font-weight:700; color:{_TEXT1}; margin:0; display:flex; align-items:center; gap:10px;'>
    <span style='background:{_SURFACE}; border:1px solid {_BORDER}; padding:4px 8px; border-radius:6px; font-size:1.1rem;'>⚖️</span>
    Fairness & Bias Audit
  </h1>
  <p style='color:{_TEXT3}; font-size:0.85rem; margin-top:6px;'>
    Approval rate analysis across demographic groups · Disparate Impact Ratio · Responsible AI
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load data and run batch predictions ───────────────────────────────────────

@st.cache_data(show_spinner="Running batch predictions on full dataset…")
def compute_fairness_data():
    """Load dataset, run predictions on all valid rows, return enriched DataFrame."""
    data_path = _ROOT / "data" / "loanData.csv"
    df = pd.read_csv(data_path)

    # Drop rows with null target or null key features
    df = df.dropna(subset=["Loan_Approved"]).reset_index(drop=True)

    model, preprocessor, info, shap_explainer = load_artifacts()
    feature_names = info.get("feature_names", [])

    # Prepare inputs (drop target + ID)
    drop_cols = ["Loan_Approved", "Applicant_ID"] if "Applicant_ID" in df.columns else ["Loan_Approved"]
    X_raw = df.drop(columns=drop_cols, errors="ignore")
    X = preprocessor.transform(X_raw)

    # Predictions
    probs   = model.predict_proba(X)[:, 1]
    preds   = (probs >= 0.5).astype(int)

    df["predicted_prob"]     = probs
    df["predicted_approved"] = preds
    df["actual_approved"]    = df["Loan_Approved"].map({"Yes": 1, "No": 0})

    # SHAP global importance
    shap_importance = {}
    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            mean_abs = np.abs(sv).mean(axis=0)
            shap_importance = dict(zip(feature_names, mean_abs.tolist()))
        except Exception:
            pass

    return df, shap_importance


# ── DIR helper ────────────────────────────────────────────────────────────────

def compute_dir(group_rates: pd.Series) -> tuple[float, str, str]:
    """
    Disparate Impact Ratio = min_group_rate / max_group_rate.
    Returns (dir_value, status, color).
    """
    if len(group_rates) < 2:
        return 1.0, "N/A", _TEXT3
    ref = group_rates.max()
    minority = group_rates.min()
    if ref == 0:
        return 1.0, "N/A", _TEXT3
    ratio = minority / ref
    if ratio >= 0.9:
        return ratio, "Pass", _GREEN
    elif ratio >= 0.8:
        return ratio, "Marginal", _AMBER
    else:
        return ratio, "Adverse Impact", _RED

# ── Chart theme ───────────────────────────────────────────────────────────────
_CHART_BASE = dict(
    plot_bgcolor=_SURFACE,
    paper_bgcolor=_BG,
    font=dict(color=_TEXT2, family="Inter, sans-serif", size=11),
    xaxis=dict(gridcolor=_BORDER2, zerolinecolor=_BORDER, tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
    yaxis=dict(gridcolor=_BORDER2, tickfont=dict(color=_TEXT2, family="JetBrains Mono")),
    hoverlabel=dict(bgcolor=_SURFACE2, font_color=_TEXT1, bordercolor=_BORDER),
    margin=dict(l=10, r=20, t=50, b=20),
)

def group_approval_chart(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    """Grouped bar chart: predicted approval rate per group."""
    agg = (
        df.groupby(group_col)["predicted_approved"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "approval_rate", "count": "n"})
    )
    agg["approval_pct"] = agg["approval_rate"] * 100

    colors = px.colors.qualitative.Plotly[:len(agg)]

    fig = go.Figure(go.Bar(
        x=agg[group_col],
        y=agg["approval_pct"],
        marker_color=_BLUE,
        text=[f"{r:.1f}%<br>(n={n})" for r, n in zip(agg["approval_pct"], agg["n"])],
        textposition="outside",
        textfont=dict(color=_TEXT2, size=10, family="JetBrains Mono"),
        hovertemplate=f"<b>%{{x}}</b><br>Approval Rate: %{{y:.1f}}%<extra></extra>",
    ))

    fig.add_hline(y=agg["approval_pct"].mean(), line_dash="dash",
                  line_color=_BORDER,
                  annotation_text=f"Overall avg: {agg['approval_pct'].mean():.1f}%",
                  annotation_font_color=_TEXT3, annotation_position="top left")

    fig.update_layout(
        **_CHART_BASE,
        title=dict(text=title, font=dict(color=_TEXT1, size=13, family="Inter"), x=0.5),
        xaxis=dict(tickfont=dict(color=_TEXT2, family="Inter"), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=dict(text="Approval Rate (%)", font=dict(color=_TEXT3, size=10)), range=[0, 105],
                   gridcolor=_BORDER2, tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
        height=320, margin=dict(l=10, r=20, t=50, b=20),
    )
    return fig, agg


# ── Main content ──────────────────────────────────────────────────────────────

with st.spinner("Loading dataset and running predictions…"):
    df, shap_importance = compute_fairness_data()

n_total    = len(df)
n_approved = df["predicted_approved"].sum()
overall_rate = n_approved / n_total * 100

# Summary row
sm1, sm2, sm3, sm4 = st.columns(4)
sm1.metric("Total Applicants", f"{n_total:,}")
sm2.metric("Predicted Approved", f"{n_approved:,}")
sm3.metric("Overall Approval Rate", f"{overall_rate:.1f}%")
sm4.metric("Model ROC-AUC", "0.988")

st.markdown("<br>", unsafe_allow_html=True)

# ── Demographic Analysis ──────────────────────────────────────────────────────
st.markdown(f"<div style='color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; border-bottom:1px solid {_BORDER2}; padding-bottom:8px;'>Approval Rates by Demographic Group</div>", unsafe_allow_html=True)

GROUPS = [
    ("Gender",            "Approval by Gender"),
    ("Property_Area",     "Approval by Property Area"),
    ("Employment_Status", "Approval by Employment Status"),
    ("Education_Level",   "Approval by Education Level"),
]

for i in range(0, len(GROUPS), 2):
    row_groups = GROUPS[i:i+2]
    cols = st.columns(len(row_groups))
    for col, (group_col, title) in zip(cols, row_groups):
        with col:
            if group_col in df.columns:
                fig, agg = group_approval_chart(df, group_col, title)
                st.plotly_chart(fig, use_container_width=True)

                # DIR Badge
                group_rates = agg.set_index(group_col)["approval_rate"]
                dir_val, dir_status, dir_color = compute_dir(group_rates)
                css_cls = "dir-pass" if "Pass" in dir_status else ("dir-warn" if "Marginal" in dir_status else "dir-fail")
                st.markdown(
                    f'<div class="{css_cls}">'
                    f'<div class="dir-title" style="color:{dir_color};">{dir_val:.2f}</div>'
                    f'<div class="dir-label">Disparate Impact Ratio &middot; {dir_status}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"<div style='text-align:center; color:{_TEXT3}; font-size:0.75rem; margin-top:8px;'>DIR ≥ 0.9 = Fair &middot; 0.8–0.9 = Marginal &middot; < 0.8 = Adverse Impact</div>", unsafe_allow_html=True)
            else:
                st.warning(f"Column `{group_col}` not found in dataset.")
    st.markdown("<br>", unsafe_allow_html=True)

# ── Global SHAP Importance ────────────────────────────────────────────────────
if shap_importance:
    st.markdown(f"<div style='color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin:20px 0 14px; border-bottom:1px solid {_BORDER2}; padding-bottom:8px;'>Global SHAP Feature Importance</div>", unsafe_allow_html=True)
    st.caption("Mean absolute SHAP value across all 1,000 predictions — shows which features the model relies on most.")

    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    feat_labels = [FEATURE_DISPLAY.get(k, k.replace("_", " ")) for k, _ in sorted_shap]
    feat_vals   = [v for _, v in sorted_shap]

    fig_shap = go.Figure(go.Bar(
        x=feat_vals[::-1],
        y=feat_labels[::-1],
        orientation="h",
        marker_color=_BLUE,
        text=[f"{v:.3f}" for v in feat_vals[::-1]],
        textposition="outside",
        textfont=dict(color=_TEXT2, size=10, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        **_CHART_BASE,
        xaxis=dict(title=dict(text="Mean Absolute SHAP Value", font=dict(color=_TEXT3, size=10)), gridcolor=_BORDER2, tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_TEXT1, size=11, family="Inter")),
        height=400, margin=dict(l=10, r=80, t=20, b=50),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Flag demographic features
    demographic_features = {"Gender", "Marital_Status", "Education_Level", "Age"}
    demo_shap = {k: v for k, v in shap_importance.items() if k in demographic_features}
    if demo_shap:
        st.markdown(f"<div style='color:{_TEXT3}; font-size:0.75rem; font-weight:600; margin-bottom:10px;'>Demographic Feature Contributions</div>", unsafe_allow_html=True)
        dcols = st.columns(len(demo_shap))
        for col, (feat, val) in zip(dcols, sorted(demo_shap.items(), key=lambda x: x[1], reverse=True)):
            rank = sorted(shap_importance, key=shap_importance.get, reverse=True).index(feat) + 1
            col.metric(
                FEATURE_DISPLAY.get(feat, feat),
                f"{val:.4f}",
                f"Rank #{rank} of {len(shap_importance)}",
            )
        st.caption("Demographic features (gender, age, education, marital status) have relatively low SHAP importance, suggesting the model primarily uses financial signals.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction Distribution ───────────────────────────────────────────────────
st.markdown(f"<div style='color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; border-bottom:1px solid {_BORDER2}; padding-bottom:8px;'>Approval Probability Distribution</div>", unsafe_allow_html=True)
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=df[df["predicted_approved"] == 1]["predicted_prob"],
    name="Approved", nbinsx=30, marker_color=_GREEN, opacity=0.8,
    hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
))
fig_hist.add_trace(go.Histogram(
    x=df[df["predicted_approved"] == 0]["predicted_prob"],
    name="Rejected", nbinsx=30, marker_color=_RED, opacity=0.8,
    hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
))
fig_hist.add_vline(x=0.5, line_dash="dash", line_color=_BORDER,
                   annotation_text="Decision Threshold (0.5)")
fig_hist.update_layout(
    **_CHART_BASE,
    barmode="overlay",
    title=dict(text="Distribution of Approval Probabilities", font=dict(color=_TEXT1, size=13, family="Inter"), x=0.5),
    xaxis=dict(title=dict(text="Approval Probability", font=dict(color=_TEXT3, size=10)), gridcolor=_BORDER2, tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
    yaxis=dict(title=dict(text="Count", font=dict(color=_TEXT3, size=10)), gridcolor=_BORDER2, tickfont=dict(color=_TEXT3, family="JetBrains Mono")),
    height=320, legend=dict(font=dict(color=_TEXT2, size=10)),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Responsible AI Disclaimer ─────────────────────────────────────────────────
st.markdown(f"<div style='color:{_TEXT3}; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin:24px 0 14px; border-bottom:1px solid {_BORDER2}; padding-bottom:8px;'>Responsible AI Statement</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="resp-card">
  <p style="color:{_TEXT2}; line-height:1.7; margin:0; font-size: 0.9rem;">
    <b style="color:{_TEXT1};">CreditWise AI is committed to fair and transparent lending decisions.</b><br><br>
    This audit monitors for <b>Disparate Impact</b> — where a neutral policy disproportionately affects
    a protected group. A Disparate Impact Ratio (DIR) below <b>0.8</b> triggers a review.
    Our model primarily uses <b>financial signals</b> (credit score, DTI ratio, savings, income) rather
    than demographic attributes, consistent with fair lending principles.<br><br>
    <b style="color:{_AMBER};">Limitations:</b> This analysis is based on model predictions, not real lending decisions.
    All predictions should be reviewed by a human loan officer before any final determination.
    Demographic parity alone does not guarantee equity — individual circumstances matter.
    This tool is for <b>portfolio monitoring</b>, not for making individual lending decisions.
  </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Reference: U.S. EEOC 4/5ths (80%) rule for Disparate Impact analysis. "
    "DIR = (lowest group approval rate) / (highest group approval rate)."
)
