import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use central data utilities
from data_utils import load_and_merge, DATA_DIR, OUT_DIR, load_power
import statsmodels.api as sm

os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":
    df = load_and_merge()

    no_null = df.dropna()

    # 1) Summary statistics
    desc = no_null.describe(include="all")

    corr = no_null.corr()
    missing = df.isna().sum()

    # Save to Excel with multiple sheets
    out_xlsx = os.path.join(OUT_DIR, "exploratory_summary.xlsx")
    with pd.ExcelWriter(out_xlsx) as writer:
        desc.to_excel(writer, sheet_name="describe")
        corr.to_excel(writer, sheet_name="correlations")
        missing.to_frame("missing_count").to_excel(writer, sheet_name="missing")

    print(f"Saved exploratory summary to {out_xlsx}")

    # 2) Heatmap of correlations
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    out_heat = os.path.join(OUT_DIR, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_heat)
    plt.close()
    print(f"Saved heatmap to {out_heat}")

    # 3) Scatterplots (temp vs power)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=no_null, x="temp", y="power")
    out_scat = os.path.join(OUT_DIR, "temp_vs_power.png")
    plt.tight_layout()
    plt.savefig(out_scat)
    plt.close()
    print(f"Saved scatter to {out_scat}")

    # 4) Population buckets mean power
    pop_means = no_null.groupby(pd.cut(no_null["Population"], 5))["power"].mean()
    out_csv = os.path.join(OUT_DIR, "population_power_means.csv")
    pop_means.to_csv(out_csv)
    print(f"Saved population means to {out_csv}")

    # 5) Linear (OLS) analysis using statsmodels
    feature_cols = [c for c in no_null.columns if c not in ("Date", "power")]
    if feature_cols:
        X = no_null[feature_cols].copy()
        y = no_null["power"].copy()

        # Add constant for intercept
        X_sm = sm.add_constant(X, has_constant="add")
        ols_model = sm.OLS(y, X_sm).fit()

        # Save textual summary
        summary_txt = os.path.join(OUT_DIR, "ols_summary.txt")
        with open(summary_txt, "w") as f:
            f.write(ols_model.summary().as_text())
        print(f"Saved OLS summary to {summary_txt}")

        # Save coefficients and p-values
        coefs = ols_model.params.rename("coef").to_frame()
        coefs["pvalue"] = ols_model.pvalues
        coefs.to_csv(os.path.join(OUT_DIR, "ols_coefficients_pvalues.csv"))
        print(
            f"Saved OLS coefficients and p-values to {os.path.join(OUT_DIR, 'ols_coefficients_pvalues.csv')}"
        )
