# Credit Card Behaviour Score Prediction

## Overview

Bank A aims to develop a forward-looking Behaviour Score—a classification model predicting whether a credit card customer will default in the following month. This project uses anonymized historical data of over 30,000 customers to build an interpretable and high‑performance credit risk model.

---

## Project Structure

```
CreditCardProject/
├── .gitignore
├── README.md
├── CreditCardBehaviourScorePred.ipynb   # Jupyter notebook with full pipeline
├── submission_22112075.csv # Final predictions
├── Datasets_final/
│   ├── train_dataset_final1.csv         # Training data 
│   └── validate_dataset_final.csv       # Validation data 
├── requirements.txt                     # Python package dependencies
└── .venv/                               # Virtual environment
```

---

## Dataset Description

* **Customer\_ID**: Unique identifier
* **Demographics**: `sex`, `education`, `marriage`, `age`, `LIMIT_BAL`
* **Payment Status**: `pay_0` to `pay_6` (last 6 months)
* **Bill Amounts**: `Bill_amt1` to `Bill_amt6`
* **Payment Amounts**: `pay_amt1` to `pay_amt6`
* **Aggregates**: `AVG_Bill_amt`, `PAY_TO_BILL_ratio`
* **Target**: `next_month_default` (1 = default, 0 = no default)

---

## Key Steps

### 1. Environment Setup

```bash
git clone https://github.com/payalkanyan/CreditCardBehPred
cd CreditCardBehPred
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis (EDA)

* Class balance and default rate (19%)
* Demographic trends (higher default among younger customers)
* Payment behaviour analysis (`pay_0` correlation)
* Advanced EDA: correlation heatmap, trends over time

### 3. Feature Engineering

* **Delay features**: `avg_delay`, `delay_count`, `max_delay`, `improvement`
* **Financial ratios**: `util_ratio`, `underpay_ratio`
* Raw aggregates: `bill_total`, `pay_total`

### 4. Model Training & Tuning

* Train/test split (80/20) with `stratify`
* Handle class imbalance via **SMOTE**
* Models compared:

  * Logistic Regression (baseline & tuned threshold)
  * Decision Tree, Random Forest
  * XGBoost, LightGBM
  * **StackingClassifier** (LogReg+RF+XGB) with meta learner
* Evaluation metrics: **F2-score** (primary), AUC-ROC, Precision, Recall
* Threshold tuning for best F2 (optimal = 0.25)


## Usage

1. Open `CreditCardBehaviourScorePred.ipynb`
2. Run all cells in order
3. Inspect results and threshold tuning
4. Generate final submission

---

## Dependencies

Listed in `requirements.txt`:

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
```

