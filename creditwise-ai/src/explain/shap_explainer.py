"""
shap_explainer.py
SHAP utilities for inference-time explanations.
Used by the FastAPI /predict endpoint.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path

SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / 'saved_models'


def create_shap_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """Create a TreeExplainer with a capped background dataset."""
    if len(X_background) > 100:
        X_background = X_background.sample(100, random_state=42)
    return shap.TreeExplainer(model, X_background)


def get_shap_values_for_instance(
    explainer: shap.TreeExplainer,
    X_instance: pd.DataFrame,
    feature_names: list,
) -> dict:
    """
    Compute SHAP values for a single prediction row.
    Returns {feature_name: shap_value}.
    """
    sv = explainer.shap_values(X_instance)

    # Handle old SHAP API returning list[class0, class1]
    if isinstance(sv, list):
        sv = sv[1]

    if sv.ndim == 2:
        sv = sv[0]  # first (only) instance

    return {name: round(float(val), 4) for name, val in zip(feature_names, sv)}


def get_top_factors(shap_dict: dict, top_n: int = 3) -> dict:
    """
    Return top N positive (approval) and top N negative (rejection) SHAP factors.
    """
    sorted_items = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    return {
        'positive': [
            {'feature': k, 'value': v}
            for k, v in sorted_items[:top_n] if v > 0
        ],
        'negative': [
            {'feature': k, 'value': v}
            for k, v in sorted_items[-top_n:] if v < 0
        ],
    }
