# 🏦 CreditWise AI

> **Production-grade loan approval intelligence platform** — built as a portfolio project by Guradesh Dhillon (3rd-year Computer Engineering, AI & Data Science specialization).

[![CI](https://github.com/your-username/creditwise-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/creditwise-ai/actions)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?logo=fastapi)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Claude AI](https://img.shields.io/badge/LLM-Claude%20Sonnet-ff6b35)

---

## 🎯 What This Project Does

CreditWise AI transforms a basic ML notebook into a fully deployable, interview-worthy AI system:

| Component | Technology | Description |
|---|---|---|
| **ML Core** | XGBoost + Optuna | 5-model comparison, 10-fold CV, 50-trial hyperparameter search |
| **Explainability** | SHAP | Per-prediction feature importance, global summary plots |
| **REST API** | FastAPI + Pydantic v2 | Validated endpoints, OpenAPI docs, CORS-ready |
| **LLM Integration** | Claude Sonnet (Anthropic) | Plain-English explanations for every decision |
| **Experiment Tracking** | MLflow | Hyperparams, metrics, model artifacts logged per run |
| **Containerisation** | Docker + Compose | One-command deployment with health checks |
| **CI/CD** | GitHub Actions | Lint + test on every push |

---

## 📦 Project Structure

```
creditwise-ai/
├── data/loanData.csv              Dataset (1001 rows, 20 columns)
├── notebooks/CreditWise_original  Original beginner notebook
├── src/
│   ├── data/preprocessing.py      Feature engineering + LoanPreprocessor class
│   ├── models/train.py            Full training pipeline (Optuna-tuned XGBoost)
│   ├── api/                       FastAPI app (main, routes, schemas)
│   ├── explain/                   SHAP + Claude LLM explainers
│   └── mlops/mlflow_logger.py     Experiment tracking
├── app/streamlit_app.py           Interactive UI dashboard
├── tests/                         pytest unit + integration tests
├── saved_models/                  Trained artifacts (gitignored)
├── Dockerfile + docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/creditwise-ai.git
cd creditwise-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

### 3. Train the Model

```bash
python src/models/train.py
```

**Expected output:**
- Prints 5-model comparison table (sorted by ROC-AUC)
- Runs 50 Optuna trials for XGBoost tuning
- Saves `saved_models/model.pkl`, `preprocessor.pkl`, `model_info.json`
- Generates SHAP plots in `saved_models/plots/`

### 4. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Visit: `http://localhost:8000/docs` for interactive API docs.

### 5. Run the Streamlit Dashboard

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

### 6. Run Tests

```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_api.py -v       # Run after training
```

---

## 🔌 API Endpoints

### `GET /api/v1/health`
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_type": "XGBoost",
  "performance": {
    "accuracy": 0.72,
    "f1": 0.71,
    "roc_auc": 0.78
  }
}
```

### `POST /api/v1/predict`

**Request:**
```json
{
  "gender": "Male",
  "marital_status": "Married",
  "dependents": 1,
  "education_level": "Graduate",
  "employment_status": "Salaried",
  "employer_category": "Private",
  "age": 35,
  "applicant_income": 10000,
  "coapplicant_income": 2000,
  "loan_amount": 20000,
  "loan_term": 60,
  "existing_loans": 1,
  "dti_ratio": 0.30,
  "savings": 15000,
  "collateral_value": 40000,
  "credit_score": 720,
  "property_area": "Urban",
  "loan_purpose": "Home"
}
```

**Response:**
```json
{
  "approved": true,
  "probability": 0.7832,
  "confidence": "High",
  "decision_text": "✅ Loan Approved",
  "shap_values": { "Credit_Score_norm": 0.142, "DTI_Ratio": -0.089, ... },
  "top_factors": [
    { "feature": "Credit_Score_norm", "value": 0.142, "direction": "positive" }
  ],
  "model_version": "1.0.0"
}
```

### `POST /api/v1/explain`
Send the application + prediction result → get plain-English explanation from Claude AI.

---

## 🐳 Docker Deployment

```bash
# Start API + MLflow UI together
docker-compose up

# API: http://localhost:8000/docs
# MLflow: http://localhost:5001
```

---

## 📊 Model Performance

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| **XGBoost (Optuna-tuned)** | **~0.78** | ~0.71 | ~0.72 |
| Random Forest | ~0.75 | ~0.68 | ~0.70 |
| Logistic Regression | ~0.73 | ~0.66 | ~0.69 |
| KNN | ~0.68 | ~0.62 | ~0.65 |
| Naive Bayes | ~0.65 | ~0.60 | ~0.63 |

*Evaluated with 10-fold Stratified Cross-Validation. Exact numbers depend on data split.*

---

## 🧠 Feature Engineering

| Feature | Formula | Purpose |
|---|---|---|
| `Total_Income` | `Applicant_Income + Coapplicant_Income` | Household capacity |
| `EMI` | `Loan_Amount / Loan_Term` | Monthly burden |
| `Balance_Income` | `Total_Income - EMI × 12` | Net annual surplus |
| `Income_to_Loan_Ratio` | `Total_Income / Loan_Amount` | Affordability |
| `Savings_to_Loan_Ratio` | `Savings / Loan_Amount` | Safety buffer |
| `Collateral_to_Loan_Ratio` | `Collateral_Value / Loan_Amount` | Security ratio |
| `Credit_Score_norm` | `(Credit_Score - 300) / 600` | Normalised score |

---

## 🛠️ Tech Stack

- **Python 3.11** | **pandas 2.2** | **numpy 1.26** | **scikit-learn 1.5**
- **XGBoost 2.1** | **SHAP 0.46** | **Optuna 3.6**
- **FastAPI 0.112** | **Pydantic v2** | **Uvicorn**
- **Anthropic Claude Sonnet** | **MLflow 2.15**
- **Docker** | **GitHub Actions**

---

## 👤 Author

**Guradesh Dhillon**  
3rd-year Computer Engineering — AI & Data Science Specialization  
Built as a portfolio project for AI/ML Engineer internship applications.
