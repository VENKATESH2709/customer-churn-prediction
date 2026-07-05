"""
Customer Churn Prediction - Training Script
Dataset: Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)
Place the downloaded CSV as: data/telco_churn.csv

This script:
1. Loads and cleans the data
2. Encodes categorical features
3. Trains Logistic Regression, Random Forest, and XGBoost
4. Compares them on Precision / Recall / F1 / ROC-AUC (not just accuracy,
   since churn is an imbalanced classification problem)
5. Saves the best model + preprocessing objects for the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

DATA_PATH = "data/telco_churn.csv"
MODEL_DIR = "models"


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Standard Telco dataset quirk: TotalCharges has blank strings for
    # customers with tenure == 0, which pandas reads as object dtype.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID is just an identifier, not predictive
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_features(df, target_col="Churn"):
    df = df.copy()
    label_encoders = {}

    # Encode target: Yes/No -> 1/0
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds))

    return {
        "model": name,
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def main():
    df = load_and_clean_data(DATA_PATH)
    df_encoded, label_encoders = encode_features(df)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_scaled, y_test))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n=== Model Comparison (sorted by F1) ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Save everything the Streamlit app needs
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")
    joblib.dump(best_model_name, f"{MODEL_DIR}/best_model_name.pkl")
    results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)

    print(f"\nSaved best model ({best_model_name}) and preprocessing objects to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
