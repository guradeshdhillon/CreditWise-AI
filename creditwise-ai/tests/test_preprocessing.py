"""
test_preprocessing.py
Unit tests for the LoanPreprocessor and data loading utilities.
Run with: pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    LoanPreprocessor, load_raw_data, encode_target, FINAL_FEATURES
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal valid DataFrame matching the real dataset schema."""
    return pd.DataFrame([{
        'Applicant_ID': 1,
        'Applicant_Income': 10000,
        'Coapplicant_Income': 2000,
        'Employment_Status': 'Salaried',
        'Age': 35,
        'Marital_Status': 'Married',
        'Dependents': 1,
        'Credit_Score': 720,
        'Existing_Loans': 1,
        'DTI_Ratio': 0.3,
        'Savings': 15000,
        'Collateral_Value': 40000,
        'Loan_Amount': 20000,
        'Loan_Term': 60,
        'Loan_Purpose': 'Home',
        'Property_Area': 'Urban',
        'Education_Level': 'Graduate',
        'Gender': 'Male',
        'Employer_Category': 'Private',
        'Loan_Approved': 'Yes',
    }])


@pytest.fixture
def preprocessor(sample_df):
    """A fitted LoanPreprocessor."""
    pp = LoanPreprocessor()
    X = sample_df.drop('Loan_Approved', axis=1)
    pp.fit(X)
    return pp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoanPreprocessor:
    def test_fit_sets_fill_values(self, sample_df):
        pp = LoanPreprocessor()
        X = sample_df.drop('Loan_Approved', axis=1)
        pp.fit(X)
        assert pp.is_fitted_
        assert 'Credit_Score' in pp.fill_values_
        assert 'Loan_Amount' in pp.fill_values_

    def test_transform_produces_correct_columns(self, preprocessor, sample_df):
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        X = preprocessor.transform(X_raw)
        for feature in FINAL_FEATURES:
            assert feature in X.columns, f"Missing feature: {feature}"

    def test_transform_drops_applicant_id(self, preprocessor, sample_df):
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        X = preprocessor.transform(X_raw)
        assert 'Applicant_ID' not in X.columns

    def test_transform_no_nulls(self, preprocessor, sample_df):
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        X = preprocessor.transform(X_raw)
        assert X.isnull().sum().sum() == 0, "Output should have no nulls"

    def test_transform_with_missing_values(self, preprocessor):
        """Rows with missing values should be imputed, not raise errors."""
        row = pd.DataFrame([{
            'Applicant_ID': 99,
            'Applicant_Income': None,
            'Coapplicant_Income': None,
            'Employment_Status': None,
            'Age': None,
            'Marital_Status': None,
            'Dependents': None,
            'Credit_Score': None,
            'Existing_Loans': None,
            'DTI_Ratio': None,
            'Savings': None,
            'Collateral_Value': None,
            'Loan_Amount': None,
            'Loan_Term': None,
            'Loan_Purpose': None,
            'Property_Area': None,
            'Education_Level': None,
            'Gender': None,
            'Employer_Category': None,
        }])
        X = preprocessor.transform(row)
        assert X.isnull().sum().sum() == 0

    def test_fit_transform_combined(self, sample_df):
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        pp = LoanPreprocessor()
        X = pp.fit_transform(X_raw)
        assert X.shape[0] == 1
        assert len(X.columns) == len(FINAL_FEATURES)

    def test_transform_requires_fit(self):
        pp = LoanPreprocessor()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            pp.transform(pd.DataFrame([{'x': 1}]))

    def test_engineered_features_present(self, preprocessor, sample_df):
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        X = preprocessor.transform(X_raw)
        for feat in ['Total_Income', 'EMI', 'Balance_Income', 'Income_to_Loan_Ratio']:
            assert feat in X.columns

    def test_credit_score_normalised(self, preprocessor, sample_df):
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        X = preprocessor.transform(X_raw)
        assert 0.0 <= X['Credit_Score_norm'].iloc[0] <= 1.0

    def test_save_and_load(self, preprocessor, tmp_path, sample_df):
        path = str(tmp_path / 'test_preprocessor.pkl')
        preprocessor.save(path)
        loaded = LoanPreprocessor.load(path)
        assert loaded.is_fitted_
        X_raw = sample_df.drop('Loan_Approved', axis=1)
        X1 = preprocessor.transform(X_raw)
        X2 = loaded.transform(X_raw)
        pd.testing.assert_frame_equal(X1, X2)


class TestEncodeTarget:
    def test_yes_maps_to_1(self):
        s = pd.Series(['Yes', 'No', 'Yes'])
        encoded = encode_target(s)
        assert list(encoded) == [1, 0, 1]

    def test_null_returns_nan(self):
        s = pd.Series(['Yes', None, 'No'])
        encoded = encode_target(s)
        assert encoded.isna().sum() == 1


class TestLoadRawData:
    def test_load_existing_file(self, tmp_path):
        csv = tmp_path / 'test.csv'
        csv.write_text("A,B\n1,2\n3,4\n")
        df = load_raw_data(str(csv))
        assert len(df) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent/path/data.csv")
