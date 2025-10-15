import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn import metrics as skm

from data_utils import load_and_merge, OUT_DIR


MODEL_PATH = os.path.join(OUT_DIR, "linear_model.pkl")


def load_data():
    """Load merged dataframe using central utility. Ensures Date and power are present."""
    df = load_and_merge(save_merged=False)
    if "Date" not in df.columns:
        raise RuntimeError("Merged data does not contain 'Date' column")
    df["Date"] = pd.to_datetime(df["Date"])
    # drop rows without target
    df = df.dropna(subset=["power"])
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run scripts/training.py first.")
        sys.exit(1)

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle.get("model")
    scaler = model_bundle.get("scaler")

    if model is None or scaler is None:
        print(f"Model bundle at {MODEL_PATH} missing 'model' or 'scaler' keys")
        sys.exit(1)

    df = load_data()

    # Prepare features (same logic as training): drop Date and power
    feature_cols = [c for c in df.columns if c not in ("Date", "power")]
    if not feature_cols:
        print("No feature columns found after merging. Aborting.")
        sys.exit(1)

    X = df[feature_cols].copy()
    y = df["power"].copy()

    # Scale features and predict
    X_sc = scaler.transform(X)
    preds = model.predict(X_sc)
    df["predicted_power"] = preds

    # --- Create a sequential train/test split (same approach as the notebook) ---
    # Use a non-shuffled split so test is the most-recent 20% of data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, stratify=None
    )

    # Scale and predict on test set only
    X_test_sc = scaler.transform(X_test)
    test_preds = model.predict(X_test_sc)

    # Compute metrics for the test set
    def compute_mase(y_true, y_pred, y_train_series):
        # MASE uses the in-sample naive forecast errors from the training set
        eps = 1e-8
        denom = np.mean(np.abs(y_train_series.values[1:] - y_train_series.values[:-1]))
        if denom < eps or np.isnan(denom):
            return np.nan
        mae = skm.mean_absolute_error(y_true, y_pred)
        return mae / denom

    mae_t = skm.mean_absolute_error(y_test, test_preds)
    mse_t = skm.mean_squared_error(y_test, test_preds)
    rmse_t = np.sqrt(mse_t)
    try:
        mape_t = skm.mean_absolute_percentage_error(y_test, test_preds) * 100.0
    except Exception:
        mape_t = (
            np.mean(
                np.abs(
                    (y_test.values - test_preds)
                    / np.maximum(np.abs(y_test.values), 1e-8)
                )
            )
            * 100.0
        )

    mase_t = compute_mase(y_test, test_preds, y_train)

    metrics_test = {
        "MAE": float(mae_t),
        "MSE": float(mse_t),
        "RMSE": float(rmse_t),
        "MAPE": float(mape_t),
        "MASE": float(mase_t) if not np.isnan(mase_t) else None,
    }

    print("Test set metrics:")
    for k, v in metrics_test.items():
        print(f"  {k}: {v}")

    # Save test metrics to CSV (append if exists)
    metrics_out = os.path.join(OUT_DIR, "metrics_test.csv")
    metrics_row = pd.DataFrame([{**{"model_path": MODEL_PATH}, **metrics_test}])
    if os.path.exists(metrics_out):
        metrics_row.to_csv(metrics_out, mode="a", header=False, index=False)
    else:
        metrics_row.to_csv(metrics_out, index=False)
    print(f"Saved test metrics to {metrics_out}")

    # Export annual results
    df["year"] = df["Date"].dt.year
    annual = df.groupby("year")[["power", "predicted_power"]].mean()
    out_csv = os.path.join(OUT_DIR, "annual_results.csv")
    annual.to_csv(out_csv)
    print(f"Saved annual results to {out_csv}")

    # Time series plot
    ts_df = df.set_index("Date")[["power", "predicted_power"]]
    ax = ts_df.plot(figsize=(14, 6))
    ax.set_ylabel("power")
    ax.set_title("Actual vs Predicted Power (all available data)")
    out_plot = os.path.join(OUT_DIR, "actual_vs_predicted_timeseries.png")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved time series plot to {out_plot}")

    # --- Test-only time series plot (actual vs predicted on test split) ---
    test_dates = df.loc[X_test.index, "Date"]
    test_plot_df = pd.DataFrame(
        {
            "Date": test_dates.values,
            "power": y_test.values,
            "predicted_power": test_preds,
        }
    )
    test_plot_df.set_index("Date", inplace=True)
    ax = test_plot_df.plot(figsize=(14, 6))
    ax.set_ylabel("power")
    ax.set_title("Actual vs Predicted Power (test set)")
    out_test_plot = os.path.join(OUT_DIR, "actual_vs_predicted_test_timeseries.png")
    plt.tight_layout()
    plt.savefig(out_test_plot)
    plt.close()
    print(f"Saved test-only time series plot to {out_test_plot}")

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(df["power"], df["predicted_power"], alpha=0.4)
    plt.xlabel("Actual power")
    plt.ylabel("Predicted power")
    plt.title("Actual vs Predicted")
    out_scatter = os.path.join(OUT_DIR, "actual_vs_predicted_scatter.png")
    plt.tight_layout()
    plt.savefig(out_scatter)
    plt.close()
    print(f"Saved scatter to {out_scatter}")

    # --- Test-only scatter ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.values, test_preds, alpha=0.4)
    plt.xlabel("Actual power (test)")
    plt.ylabel("Predicted power (test)")
    plt.title("Actual vs Predicted (test set)")
    out_test_scatter = os.path.join(OUT_DIR, "actual_vs_predicted_test_scatter.png")
    plt.tight_layout()
    plt.savefig(out_test_scatter)
    plt.close()
    print(f"Saved test-only scatter to {out_test_scatter}")


if __name__ == "__main__":
    main()
