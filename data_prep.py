import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def get_data():
    df = pd.read_csv("telco_churn.csv")

    # Target: Churn (Yes/No) -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # TotalCharges has some blank strings -> convert to numeric, drop bad rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges", "Churn"])

    # Drop customerID (not predictive)
    df = df.drop(columns=["customerID"])

    # Encode all categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=["Churn"]).values
    y = df["Churn"].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_trainval, X_test, y_trainval, y_test
