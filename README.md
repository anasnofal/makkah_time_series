# Makkah Time Series Forecasting

This repository contains code and notebooks to analyze and forecast daily power consumption for Makkah (exploratory analysis, linear regression training, evaluation and a simple 10-year forecast).

Repository structure
- `notebooks/` - Jupyter notebooks (exploration + modeling)
- `data/` - raw CSV data required by the scripts
- `scripts/` - runnable scripts
  - `data_utils.py` - central data loading & merging helpers (used by all scripts)
  - `linear_analysis.py` - exploratory analysis and OLS (statsmodels) summary
  - `training.py` - train a LinearRegression model and save model+scaler bundle
  - `result.py` - load model bundle, predict, evaluate, save plots and a 10-year forecast
- `results/` - outputs produced by the scripts (plots, CSVs, model bundles)
- `main.py` - convenience entrypoint to run the scripts from the project root

Getting started (recommended)
1) Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the pipeline (three separate steps) or the full pipeline:

```bash
# individual steps
python main.py analyze
python main.py train
python main.py result

# or run all in order (analyze -> train -> result)
python main.py all
```

What each script does
- `python main.py analyze` - produces exploratory outputs in `results/` including descriptive statistics, correlation heatmap, and an OLS summary (text and coefficients CSV). Useful to inspect relationships and p-values.
- `python main.py train` - trains the LinearRegression model on non-missing rows and saves a joblib bundle `results/linear_model.pkl` containing `{'model', 'scaler'}`.
- `python main.py result` - loads `results/linear_model.pkl`, predicts, computes test metrics (MAE, MSE, RMSE, MAPE, MASE), produces Actual vs Predicted plots (all-data and test-only), and saves a simple 10-year forecast.

Outputs and where to find them
- `results/linear_model.pkl` - saved model bundle (joblib) created by `training.py`.
- `results/metrics_test.csv` - appended test metrics from `result.py`.
- `results/*.png` - plots (time-series, scatter, test-only plots, forecast plot).
- `results/forecast_10y.csv` - simple 10-year forecast (annual averages) produced by `result.py`.
- `results/ols_summary.txt` and `results/ols_coefficients_pvalues.csv` - OLS summary and coefficients produced by `linear_analysis.py`.

Forecast assumptions (important)
- The 10-year forecast implemented in `result.py` is intentionally simple:
  - Population is extrapolated by linear growth estimated from the last available population entries (falling back to the last observed value if not enough datapoints).
  - Other covariates (temperature, GDP, etc.) are held at their most recently observed or mean values. No complex time-series forecasting was implemented for covariates.
  - Forecast outputs are annual averages; if you want daily forecasts with seasonal patterns, I can implement that.

Troubleshooting
- If pandas complains about writing Excel files, install `openpyxl`:

```bash
pip install openpyxl
```

- If `MASE` is NaN in metrics, that means the in-sample naive error (used as denominator) was zero or undefined; MASE will be skipped in that case.

Development tips / next improvements
- Add prediction intervals for the 10-year forecast using bootstrap or model-based intervals.
- Improve covariate forecasts (temperature, GDP) with ARIMA or ETS models and then feed them into the final forecast.
- Add logging (capture stdout/stderr per run) and a reproducible run manifest (timestamped outputs).

Contact / help
If you'd like any of the improvements above or want the forecast presented in a different format (daily forecasts, confidence bands, or interactive plots), tell me which and I will implement it.
