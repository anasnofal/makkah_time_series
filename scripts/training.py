import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics as skm

from data_utils import load_and_merge, OUT_DIR

MODEL_PATH = os.path.join(OUT_DIR, "linear_model.pkl")


def load_prepare():
    # load merged dataframe (power merged by Date, population by year)
    df = load_and_merge()
    # drop rows without target
    df = df.dropna(subset=["power"])
    # drop non-feature columns
    X = df.drop(columns=["power", "Date"], errors="ignore")
    y = df["power"]
    return X, y


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = load_prepare()

    # train/test split (chronological split not shuffled)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    preds = model.predict(X_test_sc)

    results = {
        "MAE": skm.mean_absolute_error(y_test, preds),
        "MSE": skm.mean_squared_error(y_test, preds),
        "RMSE": skm.root_mean_squared_error(y_test, preds),
        "MAPE": skm.mean_absolute_percentage_error(y_test, preds) * 100.0,
    }

    joblib.dump({"model": model, "scaler": sc}, MODEL_PATH)
    print(f"Saved trained model + scaler to {MODEL_PATH}")
    print("Evaluation:")
    print(results)
