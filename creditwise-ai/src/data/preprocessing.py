"""
preprocessing.py
Single source of truth for all data loading, cleaning, and feature engineering.
Adapted to the actual loanData.csv columns (20-column richer dataset).
Both training and inference must use the exact same LoanPreprocessor instance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Opt in to future pandas behavior: suppress FutureWarning about silent downcasting
pd.set_option('future.no_silent_downcasting', True)

# ── Categorical maps ──────────────────────────────────────────────────────────
GENDER_MAP = {'Male': 1, 'Female': 0}
MARITAL_STATUS_MAP = {'Married': 1, 'Single': 0}
EDUCATION_MAP = {'Graduate': 1, 'Not Graduate': 0}

EMPLOYMENT_STATUS_MAP = {
    'Salaried': 0,
    'Self-employed': 1,
    'Contract': 2,
    'Unemployed': 3,
}

EMPLOYER_CATEGORY_MAP = {
    'Private': 0,
    'Government': 1,
    'MNC': 2,
    'Business': 3,
    'Unemployed': 4,
}

PROPERTY_AREA_MAP = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}

LOAN_PURPOSE_MAP = {
    'Home': 0,
    'Car': 1,
    'Personal': 2,
    'Business': 3,
    'Education': 4,
}

# ── Feature lists ─────────────────────────────────────────────────────────────
RAW_DROP_COLS = ['Applicant_ID']

CATEGORICAL_COLS = [
    'Gender', 'Marital_Status', 'Education_Level',
    'Employment_Status', 'Employer_Category',
    'Property_Area', 'Loan_Purpose',
]

NUMERIC_COLS = [
    'Applicant_Income', 'Coapplicant_Income', 'Age',
    'Credit_Score', 'Existing_Loans', 'DTI_Ratio',
    'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term',
    'Dependents',
]

ENGINEERED_FEATURES = [
    'Total_Income', 'EMI', 'Balance_Income',
    'Income_to_Loan_Ratio', 'Savings_to_Loan_Ratio',
    'Collateral_to_Loan_Ratio', 'LoanAmount_log',
    'Total_Income_log', 'Credit_Score_norm',
]

FINAL_FEATURES = [
    # Encoded categoricals
    'Gender', 'Marital_Status', 'Dependents', 'Education_Level',
    'Employment_Status', 'Employer_Category', 'Property_Area', 'Loan_Purpose',
    # Raw numerics
    'Applicant_Income', 'Coapplicant_Income', 'Age',
    'Credit_Score', 'Existing_Loans', 'DTI_Ratio',
    'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term',
    # Engineered
    'Total_Income', 'EMI', 'Balance_Income',
    'Income_to_Loan_Ratio', 'Savings_to_Loan_Ratio',
    'Collateral_to_Loan_Ratio', 'LoanAmount_log',
    'Total_Income_log', 'Credit_Score_norm',
]


class LoanPreprocessor:
    """
    Fits on training data, transforms both train and inference data consistently.
    Save with joblib alongside the model for production inference consistency.
    """

    def __init__(self):
        self.fill_values_: dict = {}
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> 'LoanPreprocessor':
        """Compute fill values from training data only (no target leakage)."""
        self.fill_values_ = {
            'Gender': df['Gender'].mode()[0] if df['Gender'].notna().any() else 'Male',
            'Marital_Status': df['Marital_Status'].mode()[0] if df['Marital_Status'].notna().any() else 'Single',
            'Dependents': df['Dependents'].median() if df['Dependents'].notna().any() else 0,
            'Education_Level': df['Education_Level'].mode()[0] if df['Education_Level'].notna().any() else 'Graduate',
            'Employment_Status': df['Employment_Status'].mode()[0] if df['Employment_Status'].notna().any() else 'Salaried',
            'Employer_Category': df['Employer_Category'].mode()[0] if df['Employer_Category'].notna().any() else 'Private',
            'Property_Area': df['Property_Area'].mode()[0] if df['Property_Area'].notna().any() else 'Urban',
            'Loan_Purpose': df['Loan_Purpose'].mode()[0] if df['Loan_Purpose'].notna().any() else 'Personal',
            'Age': df['Age'].median() if df['Age'].notna().any() else 35,
            'Credit_Score': df['Credit_Score'].median() if df['Credit_Score'].notna().any() else 650,
            'Existing_Loans': df['Existing_Loans'].median() if df['Existing_Loans'].notna().any() else 1,
            'DTI_Ratio': df['DTI_Ratio'].median() if df['DTI_Ratio'].notna().any() else 0.35,
            'Savings': df['Savings'].median() if df['Savings'].notna().any() else 5000,
            'Collateral_Value': df['Collateral_Value'].median() if df['Collateral_Value'].notna().any() else 20000,
            'Loan_Amount': df['Loan_Amount'].median() if df['Loan_Amount'].notna().any() else 20000,
            'Loan_Term': df['Loan_Term'].mode()[0] if df['Loan_Term'].notna().any() else 60,
            'Coapplicant_Income': df['Coapplicant_Income'].median() if df['Coapplicant_Income'].notna().any() else 0,
            'Applicant_Income': df['Applicant_Income'].median() if df['Applicant_Income'].notna().any() else 8000,
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations. Returns clean feature DataFrame."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform()")

        df = df.copy()

        # Strip column name whitespace (defensive)
        df.columns = df.columns.str.strip()

        # Drop identifier columns
        for col in RAW_DROP_COLS:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # ── Fill missing values ────────────────────────────────────────────
        for col, val in self.fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val).infer_objects(copy=False)

        # ── Encode categoricals ────────────────────────────────────────────
        df['Gender'] = df['Gender'].map(GENDER_MAP).fillna(0).astype(int)
        df['Marital_Status'] = df['Marital_Status'].map(MARITAL_STATUS_MAP).fillna(0).astype(int)
        df['Education_Level'] = df['Education_Level'].map(EDUCATION_MAP).fillna(0).astype(int)
        df['Employment_Status'] = df['Employment_Status'].map(EMPLOYMENT_STATUS_MAP).fillna(0).astype(int)
        df['Employer_Category'] = df['Employer_Category'].map(EMPLOYER_CATEGORY_MAP).fillna(0).astype(int)
        df['Property_Area'] = df['Property_Area'].map(PROPERTY_AREA_MAP).fillna(1).astype(int)
        df['Loan_Purpose'] = df['Loan_Purpose'].map(LOAN_PURPOSE_MAP).fillna(2).astype(int)

        # ── Ensure numeric types ───────────────────────────────────────────
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                    self.fill_values_.get(col, 0)
                )

        # ── Feature engineering ────────────────────────────────────────────
        df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']

        # EMI: protect against division by zero
        df['EMI'] = np.where(
            df['Loan_Term'] > 0,
            df['Loan_Amount'] / df['Loan_Term'],
            df['Loan_Amount']
        )

        # Balance income (can be negative — valid info)
        df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 12)

        # Ratios — clamp to avoid inf/nan from zero denominators
        safe_loan = np.maximum(df['Loan_Amount'], 1)
        df['Income_to_Loan_Ratio'] = df['Total_Income'] / safe_loan
        df['Savings_to_Loan_Ratio'] = np.maximum(df['Savings'], 0) / safe_loan
        df['Collateral_to_Loan_Ratio'] = np.maximum(df['Collateral_Value'], 0) / safe_loan

        # Log transforms
        df['LoanAmount_log'] = np.log1p(np.maximum(df['Loan_Amount'], 0))
        df['Total_Income_log'] = np.log1p(np.maximum(df['Total_Income'], 0))

        # Normalise Credit_Score to 0-1 range (300–900 typical range)
        df['Credit_Score_norm'] = (df['Credit_Score'] - 300) / 600.0
        df['Credit_Score_norm'] = df['Credit_Score_norm'].clip(0, 1)

        # Return only final feature set (order is critical for SHAP consistency)
        available = [f for f in FINAL_FEATURES if f in df.columns]
        return df[available]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"Preprocessor saved -> {path}")

    @classmethod
    def load(cls, path: str) -> 'LoanPreprocessor':
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LoanPreprocessor, got {type(obj)}")
        return obj


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV with defensive error handling."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # defensive whitespace strip
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {csv_path}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"Missing values:\n{missing}")
    return df


def encode_target(series: pd.Series) -> pd.Series:
    """Convert Loan_Approved Yes/No to 1/0. Drops rows with null target."""
    encoded = series.map({'Yes': 1, 'No': 0})
    null_count = encoded.isna().sum()
    if null_count:
        print(f"Warning: dropping {null_count} rows with null target.")
    return encoded
