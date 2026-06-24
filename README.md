# Loan Approval Prediction

An end-to-end supervised ML pipeline that predicts whether a loan application will be approved or rejected using multiple classification algorithms.

---

## Objective

Loan approval decisions involve evaluating many applicant attributes — income, credit history, employment status, and more. This project builds a binary classification model to automate and evaluate that decision using machine learning, helping financial institutions reduce manual review time and improve consistency.

---

## Dataset

| Property | Detail |
|---|---|
| Task | Binary Classification |
| Target | `Loan_Status` (Approved / Rejected) |

**Key feature categories:**
- **Applicant info** — Gender, Marital Status, Dependents, Education, Employment
- **Financial info** — Applicant Income, Co-applicant Income, Loan Amount, Loan Term
- **Credit info** — Credit History
- **Property** — Property Area

---

## Pipeline

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of numerical and categorical features
- Correlation analysis between features and loan approval outcome
- Visualizations: count plots, histograms, heatmaps

### 2. Data Preprocessing
- Handled missing values for `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, and categorical columns
- Encoded categorical variables using Label Encoding / One-Hot Encoding

### 3. Feature Engineering
- Log transformation on skewed features (e.g. `LoanAmount`) to normalize distributions
- Created combined income feature where applicable

### 4. Model Training
Three classification algorithms trained and compared:

| Model | Description |
|---|---|
| **K-Nearest Neighbors (KNN)** | Instance-based learner using majority vote of k neighbors |
| **Logistic Regression** | Linear probabilistic classifier for binary outcomes |
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem with feature independence |

### 5. Model Evaluation
Each model evaluated using:
- **Precision** — of all predicted approvals, how many were correct
- **Recall** — of all actual approvals, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **Confusion Matrix** — breakdown of TP, TN, FP, FN

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and preprocessing |
| `scikit-learn` | Models, encoding, train-test split, metrics |
| `matplotlib` / `seaborn` | EDA visualizations |

---

## Results

All three models were trained on the same preprocessed dataset and evaluated on a held-out test set. Metrics (Precision, Recall, F1 Score) were compared across models to identify the best performer for loan approval prediction.

> *(Add your actual scores here — e.g. "Logistic Regression achieved the highest F1 Score of 0.82")*

---

## Project Structure

```
loan-approval-prediction/
│
├── loan_approval.ipynb          # Main notebook
├── loan_dataset.csv             # Dataset
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# Launch notebook
jupyter notebook loan_approval.ipynb
```
