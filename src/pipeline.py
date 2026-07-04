import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# imbalanced-learn's SMOTE may not be installed in all environments.
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    raise ImportError(
        "imblearn (imbalanced-learn) is required for SMOTE. Install it with: pip install imbalanced-learn"
    )
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")
    return df

def split_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    # GOLDEN RULE: Split FIRST before SMOTE!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Testing set:  {X_test.shape}")
    print(f"Training fraud cases: {y_train.sum()}")
    return X_train, X_test, y_train, y_test

def build_lr_pipeline():
    # StandardScaler → SMOTE → Logistic Regression
    pipeline = Pipeline([
        ('scaler',     StandardScaler()),
        ('smote',      SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', LogisticRegression(
            random_state=42, max_iter=1000, C=0.01))
    ])
    return pipeline

def build_rf_pipeline():
    # SMOTE → Random Forest (no scaler needed!)
    pipeline = Pipeline([
        ('smote',      SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', RandomForestClassifier(
            random_state=42, n_jobs=-1,
            n_estimators=100, max_depth=10))
    ])
    return pipeline

def train_models(X_train, y_train):
    print("\nTraining Logistic Regression...")
    lr = build_lr_pipeline()
    lr.fit(X_train, y_train)
    print("Logistic Regression done!")

    print("\nTraining Random Forest...")
    rf = build_rf_pipeline()
    rf.fit(X_train, y_train)
    print("Random Forest done!")

    return lr, rf

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    print("="*45)
    print("FRAUD DETECTION PIPELINE")
    print("="*45)

    df = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    lr_model, rf_model = train_models(X_train, y_train)
    save_model(rf_model, 'outputs/best_model_rf.pkl')

    print("\nPipeline complete!")