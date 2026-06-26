"""
mlflow_logger.py
Wraps MLflow tracking for the CreditWise AI training pipeline.
"""

import mlflow
import mlflow.xgboost
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
# Use a relative path — MLflow handles CWD resolution; absolute Windows paths
# trip up the URI scheme parser on Windows (no 'c' scheme registered).
TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "CreditWise-Loan-Approval"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(
    model,
    best_params: dict,
    metrics: dict,
    comparison_df,
    feature_names: list,
) -> None:
    """Log a complete training run to MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name="XGBoost_Optuna_Tuned"):
        # Hyperparameters
        clean_params = {k: v for k, v in best_params.items()
                        if k not in ('eval_metric', 'verbosity', 'random_state')}
        mlflow.log_params(clean_params)

        # Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # XGBoost model — log pre-saved pkl as a generic artifact
        # (mlflow.xgboost.log_model requires _estimator_type which may not be
        #  set on all XGBoost versions; using artifact logging is more robust)
        model_pkl = PROJECT_ROOT / 'saved_models' / 'model.pkl'
        if model_pkl.exists():
            mlflow.log_artifact(str(model_pkl), "model")

        # Comparison CSV
        csv_path = PROJECT_ROOT / 'saved_models' / 'model_comparison.csv'
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), "comparisons")

        # SHAP plots
        plots_dir = PROJECT_ROOT / 'saved_models' / 'plots'
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "shap_plots")

        # Model info JSON
        info_path = PROJECT_ROOT / 'saved_models' / 'model_info.json'
        if info_path.exists():
            mlflow.log_artifact(str(info_path), "artifacts")

        # Tags
        mlflow.set_tags({
            "model_type": "XGBoost",
            "n_features": len(feature_names),
            "tuning": "Optuna",
            "dataset": "loanData.csv",
        })

    print(f"MLflow run logged. View with: mlflow ui --port 5001 --backend-store-uri {TRACKING_URI}")
