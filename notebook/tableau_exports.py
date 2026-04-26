"""
Phase 5, Step 1 — Tableau-Ready Data Exports
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
        ML_outputs/model_results.csv
        ML_outputs/cluster_assignments.csv
        ML_outputs/forecast_results.csv

Output: tableau_exports/unicorns_main.csv
        tableau_exports/model_results.csv
        tableau_exports/cluster_assignments.csv
        tableau_exports/forecast.csv

Run:  python3 notebook/tableau_exports.py

Cleans column names, removes Python-specific types,
and formats dates for Tableau compatibility.
"""

import pandas as pd
import numpy as np
import os

os.makedirs("tableau_exports", exist_ok=True)

# ── 1. Main dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies_clean.csv", parse_dates=["date_joined"])
df["industry"] = df["industry"].replace(
    "Artificial Intelligence", "Artificial intelligence"
)

# Format date as string Tableau reads cleanly
df["date_joined"] = df["date_joined"].dt.strftime("%Y-%m-%d")

# Drop Python-internal columns Tableau doesn't need
drop_cols = ["log_valuation", "early_founder_flag",
             "month_joined", "quarter_joined", "decade_founded"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Rename to clean Tableau-friendly labels (no $ or special chars in field names)
df = df.rename(columns={
    "valuation_b":        "Valuation (B USD)",
    "funding_b":          "Funding (B USD)",
    "years_to_unicorn":   "Years to Unicorn",
    "funding_efficiency": "Funding Efficiency",
    "is_high_value":      "High Value Flag",
    "year_joined":        "Year Joined",
    "year_founded":       "Year Founded",
    "date_joined":        "Date Joined",
    "company":            "Company",
    "industry":           "Industry",
    "city":               "City",
    "country":            "Country",
    "continent":          "Continent",
})

df.to_csv("tableau_exports/unicorns_main.csv", index=False)
print(f"Saved unicorns_main.csv — {len(df):,} rows × {df.shape[1]} cols")
print(f"  Columns: {list(df.columns)}\n")

# ── 2. Model results ──────────────────────────────────────────────────────────
if os.path.exists("notebook/ml_outputs/model_results.csv"):
    model_df = pd.read_csv("ML_outputs/model_results.csv")
    model_df.columns = [c.replace("_", " ").title() for c in model_df.columns]
    model_df.to_csv("tableau_exports/model_results.csv", index=False)
    print(f"Saved model_results.csv — {len(model_df)} rows")
else:
    print("MISSING: ml_outputs/model_results.csv — run phase4_step6 first")

# ── 3. Cluster assignments ────────────────────────────────────────────────────
if os.path.exists("ml_outputs/step4_cluster_assignments.csv"):
    clust_df = pd.read_csv("ml_outputs/step4_cluster_assignments.csv")
    clust_df = clust_df.rename(columns={
        "company":          "Company",
        "industry":         "Industry",
        "country":          "Country",
        "continent":        "Continent",
        "valuation_b":      "Valuation (B USD)",
        "funding_b":        "Funding (B USD)",
        "years_to_unicorn": "Years to Unicorn",
        "cluster":          "Cluster ID",
        "archetype":        "Cluster Archetype",
    })
    clust_df.to_csv("tableau_exports/cluster_assignments.csv", index=False)
    print(f"Saved cluster_assignments.csv — {len(clust_df):,} rows")
else:
    print("MISSING: ml_outputs/step4_cluster_assignments.csv — run phase4_step4 first")

# ── 4. Forecast ───────────────────────────────────────────────────────────────
if os.path.exists("ml_outputs/step5_forecast_results.csv"):
    fcast_df = pd.read_csv("ml_outputs/step5_forecast_results.csv",
                            parse_dates=["month"])
    fcast_df["month"] = fcast_df["month"].dt.strftime("%Y-%m-%d")
    fcast_df = fcast_df.rename(columns={
        "month":                "Month",
        "predicted_unicorns":   "Predicted Unicorns",
        "lower_90":             "Lower 90%",
        "upper_90":             "Upper 90%",
        "trend":                "Trend",
    })
    # Add a flag so Tableau can colour history vs forecast differently
    last_actual = "2022-04-01"
    fcast_df["Is Forecast"] = (fcast_df["Month"] > last_actual).astype(int)
    fcast_df.to_csv("tableau_exports/forecast.csv", index=False)
    print(f"Saved forecast.csv — {len(fcast_df)} rows")
else:
    print("MISSING: ml_outputs/step5_forecast_results.csv — run phase4_step5 first")

print("\nAll Tableau exports saved to: tableau_exports/")
print("Connect these as separate data sources in Tableau.")