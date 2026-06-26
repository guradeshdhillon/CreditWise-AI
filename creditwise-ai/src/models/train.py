"""
train.py
Model training pipeline for CreditWise AI.
Trains 5 classifiers, compares with 10-fold StratifiedKFold CV,
tunes XGBoost with Optuna, and saves all production artifacts.

Run directly:
    python src/models/train.py
"""

import os
import sys
import json
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether called as a script or imported
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.preprocessing import (
    LoanPreprocessor, load_raw_data, encode_target, FINAL_FEATURES
)

RANDOM_STATE = 42
N_SPLITS = 10
SAVED_MODELS_DIR = _PROJECT_ROOT / 'saved_models'
SAVED_MODELS_DIR.mkdir(exist_ok=True)

SCORING = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}


# ── Baseline models ───────────────────────────────────────────────────────────

def get_baseline_models() -> dict:
    return {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'LogisticRegression': LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, C=1.0
        ),
        'NaiveBayes': GaussianNB(),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run StratifiedKFold CV on all models. Returns sorted comparison DataFrame."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in get_baseline_models().items():
        print(f"  Evaluating {name}...")
        try:
            scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            results.append({
                'Model': name,
                'Accuracy': round(scores['test_accuracy'].mean(), 4),
                'Precision': round(scores['test_precision'].mean(), 4),
                'Recall': round(scores['test_recall'].mean(), 4),
                'F1': round(scores['test_f1'].mean(), 4),
                'ROC_AUC': round(scores['test_roc_auc'].mean(), 4),
                'F1_Std': round(scores['test_f1'].std(), 4),
            })
        except Exception as e:
            print(f"  WARNING: {name} failed -> {e}. Skipping.")

    results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
    return results_df


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_xgboost_with_optuna(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
    """Hyperparameter tuning for XGBoost using Optuna. Returns best params."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'eval_metric': 'logloss',
            'random_state': RANDOM_STATE,
            'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_validate(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best ROC-AUC (Optuna): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def train_final_model(X: pd.DataFrame, y: pd.Series, best_params: dict) -> xgb.XGBClassifier:
    """Train final XGBoost on full dataset with tuned params."""
    params = {
        **best_params,
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'verbosity': 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def save_artifacts(
    model: xgb.XGBClassifier,
    preprocessor: LoanPreprocessor,
    metrics: dict,
    best_params: dict,
    feature_names: list,
) -> None:
    """Save model, preprocessor, and metadata for production inference."""
    joblib.dump(model, SAVED_MODELS_DIR / 'model.pkl')
    preprocessor.save(str(SAVED_MODELS_DIR / 'preprocessor.pkl'))

    artifact = {
        'model_version': '1.0.0',
        'model_type': 'XGBoost',
        'feature_names': feature_names,
        'best_params': best_params,
        'performance': metrics,
        'training_date': str(pd.Timestamp.now().date()),
    }
    with open(SAVED_MODELS_DIR / 'model_info.json', 'w') as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifacts saved to {SAVED_MODELS_DIR}/")
    print(f"  model.pkl | preprocessor.pkl | model_info.json")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_training_pipeline(
    data_path: str | None = None,
    n_optuna_trials: int = 50,
) -> tuple:
    """Full end-to-end training pipeline. Returns (model, preprocessor, metrics)."""
    if data_path is None:
        data_path = str(_PROJECT_ROOT / 'data' / 'loanData.csv')

    print("=" * 60)
    print("  CREDITWISE AI -- TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading data...")
    df = load_raw_data(data_path)

    # 2. Separate target before preprocessing (prevent leakage)
    print("\n[2/6] Separating features and target...")
    if 'Loan_Approved' not in df.columns:
        raise ValueError("Column 'Loan_Approved' not found in dataset.")
    y_raw = encode_target(df['Loan_Approved'])
    # Drop rows with missing target
    valid_mask = y_raw.notna()
    df = df[valid_mask].reset_index(drop=True)
    y = y_raw[valid_mask].astype(int).reset_index(drop=True)

    X_raw = df.drop('Loan_Approved', axis=1)
    print(f"  Valid samples: {len(y)} | Approved: {y.sum()} | Rejected: {(y==0).sum()}")

    # 3. Fit preprocessor
    print("\n[3/6] Fitting preprocessor on full training data...")
    preprocessor = LoanPreprocessor()
    X = preprocessor.fit_transform(X_raw)
    print(f"  Feature matrix: {X.shape}")

    # 4. Compare models
    print("\n[4/6] Comparing 5 models (10-fold Stratified CV)...")
    comparison_df = compare_models(X, y)
    print("\nModel Comparison (sorted by ROC-AUC):")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(SAVED_MODELS_DIR / 'model_comparison.csv', index=False)

    # 5. Tune XGBoost
    print(f"\n[5/6] Tuning XGBoost with Optuna ({n_optuna_trials} trials)...")
    best_params = tune_xgboost_with_optuna(X, y, n_trials=n_optuna_trials)

    # 6. Train final model + get CV metrics
    print("\n[6/6] Training final model on full dataset...")
    final_model = train_final_model(X, y, best_params)

    # Final CV evaluation
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    eval_params = {**best_params, 'eval_metric': 'logloss',
                   'random_state': RANDOM_STATE, 'verbosity': 0}
    final_scores = cross_validate(
        xgb.XGBClassifier(**eval_params), X, y,
        cv=cv, scoring=SCORING
    )
    final_metrics = {
        metric: round(float(final_scores[f'test_{metric}'].mean()), 4)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    }
    print(f"\nFinal CV Performance:\n{json.dumps(final_metrics, indent=2)}")

    # Save artifacts
    feature_names = list(X.columns)
    save_artifacts(final_model, preprocessor, final_metrics, best_params, feature_names)

    # ── SHAP background data ───────────────────────────────────────────────
    print("\nGenerating SHAP explainer...")
    try:
        import shap
        background = X.sample(min(100, len(X)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(final_model, background)
        joblib.dump(explainer, SAVED_MODELS_DIR / 'shap_explainer.pkl')
        background.to_parquet(SAVED_MODELS_DIR / 'shap_background.parquet', index=False)
        print("  SHAP explainer saved.")

        # Global plots
        _generate_shap_plots(final_model, X, feature_names)
    except Exception as e:
        print(f"  WARNING: SHAP setup failed (non-blocking): {e}")

    # ── MLflow logging ─────────────────────────────────────────────────────
    try:
        from src.mlops.mlflow_logger import log_training_run
        log_training_run(final_model, best_params, final_metrics, comparison_df, feature_names)
    except Exception as e:
        print(f"  WARNING: MLflow logging failed (non-blocking): {e}")

    print("\n[OK] Training pipeline complete!")
    return final_model, preprocessor, final_metrics


def _generate_shap_plots(model, X: pd.DataFrame, feature_names: list) -> None:
    """Generate global SHAP summary plots."""
    try:
        import shap
        import matplotlib.pyplot as plt
        plots_dir = SAVED_MODELS_DIR / 'plots'
        plots_dir.mkdir(exist_ok=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Summary dot plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(plots_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Bar plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(plots_dir / 'shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  SHAP plots saved to {plots_dir}/")
    except Exception as e:
        print(f"  WARNING: SHAP plot generation failed: {e}")


if __name__ == '__main__':
    run_training_pipeline()
