# Fraud Detection Pipeline — Supervised Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Random Forest](https://img.shields.io/badge/Random%20Forest-ROC--AUC%3D0.98-brightgreen)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced%20Data-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Project%202-orange)

---

## Project Overview

This is **Project 2** of the DecodeLabs Industrial Training Kit (Batch 2026).
This project builds a production-grade fraud detection pipeline on the
Credit Card Fraud dataset — handling extreme class imbalance using SMOTE,
training Logistic Regression and Random Forest classifiers, and evaluating
with Precision, Recall, and ROC-AUC — completely discarding misleading
Accuracy metrics as per the Zero-Leakage Protocol.

---

## The Problem — Why Accuracy is Useless Here

| Scenario | Accuracy | Fraud Caught |
|----------|----------|--------------|
| Predict everything as legitimate | 99.83% | 0% |
| Our Random Forest model | N/A | 88.78% |

A model that predicts every transaction as legitimate achieves
99.83% accuracy while catching zero fraud — causing catastrophic
financial loss. Accuracy is completely discarded in this project.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Credit Card Fraud Detection |
| Total transactions | 284,807 |
| Legitimate | 284,315 (99.83%) |
| Fraudulent | 492 (0.17%) |
| Features | 30 (V1-V28 PCA + Time + Amount) |
| Target | Class (0=Legitimate, 1=Fraud) |

---

## Project Structure
fraud-detection/
│
├── data/
│   └── creditcard.csv          ← raw dataset (gitignored)
│
├── notebooks/
│   └── fraud_detection.ipynb   ← full pipeline notebook
│
├── src/
│   ├── pipeline.py             ← train both models
│   ├── evaluate.py             ← metrics & ROC curves
│   └── predict.py              ← load model & predict
│
├── outputs/
│   ├── best_model_rf.pkl               ← saved Random Forest
│   ├── fraud_model_comparison.csv      ← results table
│   ├── roc_curve.png                   ← ROC curve plot
│   ├── Logistic_Regression_confusion.png
│   └── Random_Forest_confusion.png
│
├── .gitignore
└── README.md

---

## Pipeline Architecture — Zero Leakage Protocol

Raw Imbalanced Data (99.83% legitimate / 0.17% fraud)
↓
Stratified Train/Test Split (80/20)
↓ (SMOTE applied ONLY inside training fold!)
imblearn Pipeline Loop:
├── StandardScaler (Logistic Regression only)
├── SMOTE — synthetic minority oversampling
└── Classifier (LR or Random Forest)
↓
GridSearchCV — 5-fold cross validation
↓
Final Evaluation on untouched test set
(Precision, Recall, F1, ROC-AUC — NO Accuracy!)

---

## Why imblearn Pipeline — NOT sklearn Pipeline

| | sklearn Pipeline | imblearn Pipeline |
|--|-----------------|-------------------|
| SMOTE support | Crashes or ignored | Natively supported |
| Data leakage | Risk of leakage | Zero leakage |
| Resampling scope | Entire dataset | Training fold only |

**Golden Rule: NEVER apply SMOTE before Train/Test Split!**
Applying SMOTE before splitting means testing on synthetic data
that already knows the training answers — destroying model validity.

---

## Model Results (NO Accuracy!)

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Precision | 0.0591 | **0.5210** |
| Recall | 0.9184 | 0.8878 |
| F1 Score | 0.1110 | **0.6566** |
| ROC-AUC | 0.9719 | **0.9849** |

**Winner: Random Forest** — ROC-AUC = 0.9849

---

## Confusion Matrix Analysis

### Logistic Regression:
| | Predicted Legitimate | Predicted Fraud |
|--|---------------------|----------------|
| Actual Legitimate | 55,431 | 1,433 |
| Actual Fraud | 8 | 90 |

High Recall (0.9184) but low Precision — flags too many
legitimate transactions as fraud (1,433 false alarms!)

### Random Forest:
| | Predicted Legitimate | Predicted Fraud |
|--|---------------------|----------------|
| Actual Legitimate | 56,784 | 80 |
| Actual Fraud | 11 | 87 |

Better balance — only 80 false alarms while catching
87/98 fraud cases with ROC-AUC of 0.9849!

---

## Best Model Parameters

| Parameter | Value |
|-----------|-------|
| Model | Random Forest |
| SMOTE k_neighbors | 5 |
| n_estimators | 100 |
| max_depth | 10 |
| Scoring metric | Recall |
| Cross validation | 3-fold |

---

## How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/mehakchanna04-coder/fraud-detection.git
cd fraud-detection
```

**2. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install imbalanced-learn jupyter
```

**3. Add dataset:**
- Download `creditcard.csv` from Kaggle
- Place it in the `data/` folder

**4. Run the pipeline:**
```bash
python src/pipeline.py
```

**5. Evaluate models:**
```bash
python src/evaluate.py
```

**6. Make predictions:**
```bash
python src/predict.py
```

**7. Or open the notebook:**
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## Key Learnings

- Accuracy is a completely misleading metric for imbalanced datasets
- SMOTE creates synthetic samples via interpolation — never cloning
- SMOTE must ONLY be applied inside the training fold — never before splitting
- `imblearn.pipeline.Pipeline` is mandatory for zero-leakage resampling
- Random Forest outperforms Logistic Regression on non-linear fraud patterns
- ROC-AUC of 0.98 means the model can separate fraud from legitimate
  transactions with 98% confidence

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Models, metrics, GridSearchCV |
| Imbalanced-learn | SMOTE, imblearn Pipeline |
| Matplotlib / Seaborn | Visualization |
| Pickle | Model serialization |
| Google Colab | Development environment |
| Git + GitHub | Version control |

---

## Portfolio

**Project 1:** [house-price-ds](https://github.com/mehakchanna04-coder/house-price-ds)
— Advanced EDA & Feature Engineering

**Project 2:** [fraud-detection](https://github.com/mehakchanna04-coder/fraud-detection)
— Supervised Learning & Fraud Detection Pipeline

---

## Author

**Mehak Channa**
DecodeLabs Industrial Training — Batch 2026
GitHub: [@mehakchanna04-coder](https://github.com/mehakchanna04-coder)

---

*"Identify hidden anomalies and protect financial ecosystems
through pure machine learning logic."*
*— DecodeLabs Project 2*







