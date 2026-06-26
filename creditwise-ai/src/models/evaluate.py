"""
evaluate.py
SHAP explainability utilities for the trained XGBoost model.

Provides:
  - Global SHAP summary / bar plots (run once after training)
  - Per-instance SHAP explanation (used by the API at inference time)
  - Convenience loaders for model and preprocessor

Run standalone after training:
    python src/models/evaluate.py
"""

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Allow running as a script from the project root
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

SAVED_MODELS_DIR = _PROJECT_ROOT / "saved_models"
PLOTS_DIR = SAVED_MODELS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_model_and_preprocessor():
    """Load the trained model and fitted preprocessor from saved_models/."""
    model_path = SAVED_MODELS_DIR / "model.pkl"
    preprocessor_path = SAVED_MODELS_DIR / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.pkl not found in {SAVED_MODELS_DIR}. "
            "Run 'python src/models/train.py' first."
        )
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"preprocessor.pkl not found in {SAVED_MODELS_DIR}. "
            "Run 'python src/models/train.py' first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


# ── SHAP Explainer creation ────────────────────────────────────────────────────

def create_shap_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Create a TreeExplainer for XGBoost with a capped background dataset.

    Args:
        model:         Trained XGBoost classifier.
        X_background:  Sample of the training set (max 100 rows for efficiency).

    Returns:
        A fitted shap.TreeExplainer instance.
    """
    if len(X_background) > 100:
        X_background = X_background.sample(100, random_state=42)
    explainer = shap.TreeExplainer(model, X_background)
    return explainer


def save_shap_explainer(
    explainer: shap.TreeExplainer,
    X_background: pd.DataFrame,
) -> None:
    """Persist the SHAP explainer and background data for inference use."""
    joblib.dump(explainer, SAVED_MODELS_DIR / "shap_explainer.pkl")
    X_background.to_parquet(SAVED_MODELS_DIR / "shap_background.parquet", index=False)
    print(
        f"SHAP explainer  → {SAVED_MODELS_DIR / 'shap_explainer.pkl'}\n"
        f"Background data → {SAVED_MODELS_DIR / 'shap_background.parquet'}"
    )


# ── Per-instance SHAP ─────────────────────────────────────────────────────────

def get_shap_values_for_instance(
    explainer: shap.TreeExplainer,
    X_instance: pd.DataFrame,
    feature_names: list,
) -> dict:
    """
    Compute SHAP values for a single prediction row.

    Returns:
        dict mapping feature_name → shap_value (rounded to 4 decimal places).
    """
    sv = explainer.shap_values(X_instance)

    # Old SHAP API returns list[class_0_vals, class_1_vals]
    if isinstance(sv, list):
        sv = sv[1]

    # Shape (1, n_features) → (n_features,)
    if sv.ndim == 2:
        sv = sv[0]

    return {name: round(float(val), 4) for name, val in zip(feature_names, sv)}


def get_top_factors(shap_dict: dict, top_n: int = 3) -> dict:
    """
    Return top N positive (approval) and top N negative (rejection) factors.

    Returns:
        {
            'positive': [{'feature': str, 'value': float}, ...],
            'negative': [{'feature': str, 'value': float}, ...],
        }
    """
    sorted_items = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    return {
        "positive": [
            {"feature": k, "value": v}
            for k, v in sorted_items[:top_n]
            if v > 0
        ],
        "negative": [
            {"feature": k, "value": v}
            for k, v in sorted_items[-top_n:]
            if v < 0
        ],
    }


# ── Global SHAP plots ─────────────────────────────────────────────────────────

def generate_global_shap_plots(
    model,
    X: pd.DataFrame,
    feature_names: list,
) -> None:
    """
    Generate and save two global SHAP summary plots to saved_models/plots/.

      - shap_summary.png  — dot/beeswarm plot showing value distribution per feature
      - shap_bar.png      — horizontal bar plot of mean |SHAP| per feature

    Args:
        model:         Trained XGBoost classifier.
        X:             Full preprocessed feature matrix (used to compute SHAP values).
        feature_names: Ordered list of feature names matching X columns.
    """
    print("Computing global SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Normalise SHAP output (old API returns list, new API returns array directly)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── Dot / beeswarm summary plot ───────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP Feature Importance — CreditWise AI", fontsize=14, pad=12)
    plt.tight_layout()
    out_path = PLOTS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Bar plot (mean |SHAP|) ────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("Mean Absolute SHAP Values — CreditWise AI", fontsize=14, pad=12)
    plt.tight_layout()
    out_path = PLOTS_DIR / "shap_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    print(f"\nAll SHAP plots saved to {PLOTS_DIR}/")


# ── Standalone entry point ────────────────────────────────────────────────────

def run_evaluation() -> None:
    """
    Standalone evaluation runner.
    Loads trained artifacts, regenerates SHAP plots, and prints a summary.
    """
    import json

    print("=" * 60)
    print("  CREDITWISE AI — POST-TRAINING EVALUATION")
    print("=" * 60)

    # Load artifacts
    print("\n[1/3] Loading trained artifacts...")
    model, preprocessor = load_model_and_preprocessor()

    info_path = SAVED_MODELS_DIR / "model_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"model_info.json not found in {SAVED_MODELS_DIR}.")
    with open(info_path) as fh:
        model_info = json.load(fh)

    feature_names = model_info.get("feature_names", [])
    print(f"  Model type    : {model_info.get('model_type', 'unknown')}")
    print(f"  Model version : {model_info.get('model_version', 'unknown')}")
    print(f"  Trained on    : {model_info.get('training_date', 'unknown')}")
    print(f"  Features      : {len(feature_names)}")

    perf = model_info.get("performance", {})
    print("\n  Cross-Validation Performance:")
    for metric, value in perf.items():
        print(f"    {metric.upper():<12}: {value:.4f}")

    # Load background data for SHAP
    print("\n[2/3] Loading SHAP background data...")
    bg_path = SAVED_MODELS_DIR / "shap_background.parquet"
    if bg_path.exists():
        X_background = pd.read_parquet(bg_path)
        print(f"  Background shape: {X_background.shape}")
    else:
        print("  WARNING: shap_background.parquet not found. Run training first.")
        return

    # Re-generate global plots
    print("\n[3/3] Generating global SHAP plots...")
    generate_global_shap_plots(model, X_background, feature_names)

    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
