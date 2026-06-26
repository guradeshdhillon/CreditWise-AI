"""
test_models.py
Unit tests for the CreditWise AI model training pipeline.

Tests:
  - LoanPreprocessor integration with training functions
  - compare_models() produces a valid DataFrame
  - train_final_model() returns a fitted XGBoost classifier
  - save / load model artifacts round-trip
  - evaluate.py SHAP helpers (get_shap_values_for_instance, get_top_factors)

Run with:
    pytest tests/test_models.py -v
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import LoanPreprocessor, FINAL_FEATURES, encode_target


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mini_raw_df():
    """
    A small synthetic dataset (20 rows) that mirrors the real loanData.csv schema.
    Used to test training logic without loading the full dataset.
    """
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame({
        "Applicant_ID":     [f"A{i:03d}" for i in range(n)],
        "Gender":           rng.choice(["Male", "Female"], n).tolist(),
        "Marital_Status":   rng.choice(["Married", "Single"], n).tolist(),
        "Dependents":       rng.integers(0, 4, n).tolist(),
        "Education_Level":  rng.choice(["Graduate", "Not Graduate"], n).tolist(),
        "Employment_Status": rng.choice(["Salaried", "Self-employed", "Contract"], n).tolist(),
        "Employer_Category": rng.choice(["Private", "Government", "MNC"], n).tolist(),
        "Age":              rng.integers(22, 60, n).tolist(),
        "Applicant_Income": rng.integers(5000, 50000, n).tolist(),
        "Coapplicant_Income": rng.integers(0, 20000, n).tolist(),
        "Loan_Amount":      rng.integers(5000, 100000, n).tolist(),
        "Loan_Term":        rng.choice([12, 24, 36, 60, 120, 240], n).tolist(),
        "Existing_Loans":   rng.integers(0, 5, n).tolist(),
        "DTI_Ratio":        rng.uniform(0.05, 0.70, n).tolist(),
        "Savings":          rng.integers(0, 50000, n).tolist(),
        "Collateral_Value": rng.integers(0, 200000, n).tolist(),
        "Credit_Score":     rng.integers(400, 850, n).tolist(),
        "Property_Area":    rng.choice(["Urban", "Semiurban", "Rural"], n).tolist(),
        "Loan_Purpose":     rng.choice(["Home", "Car", "Personal"], n).tolist(),
        "Loan_Approved":    rng.choice(["Yes", "No"], n).tolist(),
    })


@pytest.fixture(scope="module")
def mini_X_y(mini_raw_df):
    """Preprocessed (X, y) pair from the synthetic dataset."""
    y_raw = encode_target(mini_raw_df["Loan_Approved"])
    valid = y_raw.notna()
    df_valid = mini_raw_df[valid].reset_index(drop=True)
    y = y_raw[valid].astype(int).reset_index(drop=True)

    X_raw = df_valid.drop("Loan_Approved", axis=1)
    pp = LoanPreprocessor()
    X = pp.fit_transform(X_raw)
    return X, y, pp


# ── Tests: preprocessing integration ─────────────────────────────────────────

class TestPreprocessingIntegration:
    def test_mini_dataset_produces_correct_shape(self, mini_X_y):
        X, y, _ = mini_X_y
        assert X.shape[0] == len(y), "Row counts must match"
        assert X.shape[1] == len(FINAL_FEATURES), f"Expected {len(FINAL_FEATURES)} features"

    def test_mini_dataset_no_nulls(self, mini_X_y):
        X, _, _ = mini_X_y
        assert X.isnull().sum().sum() == 0, "Preprocessed features must have no nulls"

    def test_target_binary(self, mini_X_y):
        _, y, _ = mini_X_y
        assert set(y.unique()).issubset({0, 1}), "Target must be binary 0/1"


# ── Tests: model training ─────────────────────────────────────────────────────

class TestModelTraining:
    def test_train_final_model_returns_fitted_model(self, mini_X_y):
        """train_final_model should return a fitted XGBoost classifier."""
        import xgboost as xgb
        from src.models.train import train_final_model

        X, y, _ = mini_X_y
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
        }
        model = train_final_model(X, y, params)
        assert isinstance(model, xgb.XGBClassifier)

        # Should be able to predict on same data
        preds = model.predict_proba(X)
        assert preds.shape == (len(y), 2)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_compare_models_returns_dataframe(self, mini_X_y):
        """compare_models should return a DataFrame with at least one row."""
        from src.models.train import compare_models

        X, y, _ = mini_X_y
        # Use only KNN and LogReg to keep tests fast
        import unittest.mock as mock
        import src.models.train as train_module

        minimal_models = {
            "KNN": __import__("sklearn.neighbors", fromlist=["KNeighborsClassifier"]).KNeighborsClassifier(n_neighbors=3),
        }
        with mock.patch.object(train_module, "get_baseline_models", return_value=minimal_models):
            result_df = compare_models(X, y)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= 1
        assert "Model" in result_df.columns
        assert "ROC_AUC" in result_df.columns

    def test_compare_models_sorted_by_roc_auc(self, mini_X_y):
        """compare_models output must be sorted descending by ROC_AUC."""
        from src.models.train import compare_models
        import unittest.mock as mock
        import src.models.train as train_module
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression

        minimal_models = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "LR": LogisticRegression(max_iter=200, random_state=42),
        }
        with mock.patch.object(train_module, "get_baseline_models", return_value=minimal_models):
            result_df = compare_models(X=mini_X_y[0], y=mini_X_y[1])

        roc_values = result_df["ROC_AUC"].tolist()
        assert roc_values == sorted(roc_values, reverse=True), "Results must be descending by ROC_AUC"


# ── Tests: artifact save / load ───────────────────────────────────────────────

class TestArtifactPersistence:
    def test_model_save_and_load(self, mini_X_y, tmp_path):
        """Model saved with joblib must load and produce identical predictions."""
        import joblib
        import xgboost as xgb
        from src.models.train import train_final_model

        X, y, _ = mini_X_y
        model = train_final_model(X, y, {"n_estimators": 10, "max_depth": 3})

        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)

        np.testing.assert_array_almost_equal(
            model.predict_proba(X),
            loaded.predict_proba(X),
            decimal=6,
            err_msg="Loaded model must produce identical probabilities",
        )

    def test_preprocessor_save_and_load(self, mini_X_y, tmp_path, mini_raw_df):
        """Saved preprocessor must produce identical output when reloaded."""
        _, _, pp = mini_X_y
        pp_path = str(tmp_path / "preprocessor.pkl")
        pp.save(pp_path)
        loaded_pp = LoanPreprocessor.load(pp_path)

        X_raw = mini_raw_df.drop("Loan_Approved", axis=1)
        pd.testing.assert_frame_equal(
            pp.transform(X_raw),
            loaded_pp.transform(X_raw),
        )

    def test_model_info_json_structure(self):
        """
        If model_info.json exists, verify it has all required keys.
        Skips gracefully if training has not been run yet.
        """
        info_path = Path("saved_models") / "model_info.json"
        if not info_path.exists():
            pytest.skip("model_info.json not found — run training first.")

        with open(info_path) as fh:
            info = json.load(fh)

        for key in ("model_version", "model_type", "feature_names", "performance"):
            assert key in info, f"model_info.json missing required key: '{key}'"

        assert "roc_auc" in info["performance"], "performance dict must contain roc_auc"
        assert info["performance"]["roc_auc"] > 0.0


# ── Tests: evaluate.py SHAP helpers ──────────────────────────────────────────

class TestEvaluateSHAPHelpers:
    def test_get_top_factors_positive_and_negative(self):
        """get_top_factors should split correctly into positive and negative."""
        from src.models.evaluate import get_top_factors

        shap_dict = {
            "Credit_Score": 0.25,
            "DTI_Ratio": -0.18,
            "Savings": 0.10,
            "Existing_Loans": -0.05,
            "Age": 0.02,
        }
        result = get_top_factors(shap_dict, top_n=3)

        assert "positive" in result and "negative" in result
        for item in result["positive"]:
            assert item["value"] > 0
        for item in result["negative"]:
            assert item["value"] < 0

    def test_get_top_factors_respects_top_n(self):
        """get_top_factors should return at most top_n items per direction."""
        from src.models.evaluate import get_top_factors

        shap_dict = {f"f{i}": float(i - 5) for i in range(10)}
        result = get_top_factors(shap_dict, top_n=2)
        assert len(result["positive"]) <= 2
        assert len(result["negative"]) <= 2

    def test_get_top_factors_all_positive(self):
        """If all SHAP values are positive, negative list should be empty."""
        from src.models.evaluate import get_top_factors

        shap_dict = {"a": 0.5, "b": 0.3, "c": 0.1}
        result = get_top_factors(shap_dict, top_n=3)
        assert result["negative"] == []

    def test_get_shap_values_returns_dict_with_feature_names(self, mini_X_y):
        """get_shap_values_for_instance must return a dict keyed by feature names."""
        import xgboost as xgb
        import shap as _shap
        from src.models.evaluate import (
            create_shap_explainer,
            get_shap_values_for_instance,
        )
        from src.models.train import train_final_model

        X, y, _ = mini_X_y
        model = train_final_model(X, y, {"n_estimators": 10, "max_depth": 3})
        explainer = create_shap_explainer(model, X)

        single_row = X.iloc[[0]]
        result = get_shap_values_for_instance(explainer, single_row, list(X.columns))

        assert isinstance(result, dict)
        assert set(result.keys()) == set(X.columns)
        for val in result.values():
            assert isinstance(val, float)

    def test_create_shap_explainer_caps_background(self, mini_X_y):
        """create_shap_explainer must not raise even when background < 100 rows."""
        import shap as _shap
        from src.models.evaluate import create_shap_explainer
        from src.models.train import train_final_model

        X, y, _ = mini_X_y
        model = train_final_model(X, y, {"n_estimators": 10, "max_depth": 3})
        # Pass fewer than 100 rows — should not raise
        explainer = create_shap_explainer(model, X)
        assert explainer is not None
