"""
Phase 2a — Data Cleaning & Feature Engineering

Input:  unicorn_companies.csv  (raw)
Output: unicorn_clean.csv      (cleaned, feature-rich)

Run: python3 notebook/clean.py | tee report/cleanReport.txt
"""

import pandas as pd
import numpy as np

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies.csv")
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ── 2. Rename columns (snake_case, no special chars) ─────────────────────────
df.columns = [
    "company", "valuation_b", "date_joined",
    "industry", "city", "country",
    "continent", "year_founded", "funding_b",
]

# ── 3. Parse dates ────────────────────────────────────────────────────────────
# Raw format is DD/MM/YYYY
df["date_joined"] = pd.to_datetime(df["date_joined"], dayfirst=True, errors="coerce")

n_bad_dates = df["date_joined"].isna().sum()
print(f"Date parse issues: {n_bad_dates} rows")

df["year_joined"]  = df["date_joined"].dt.year
df["month_joined"] = df["date_joined"].dt.month
df["quarter_joined"] = df["date_joined"].dt.quarter

# ── 4. Fix data types ─────────────────────────────────────────────────────────
df["valuation_b"] = pd.to_numeric(df["valuation_b"], errors="coerce")
df["funding_b"]   = pd.to_numeric(df["funding_b"],   errors="coerce")
df["year_founded"] = pd.to_numeric(df["year_founded"], errors="coerce")

# ── 5. Handle outliers ────────────────────────────────────────────────────────
# year_founded: companies founded before 1990 are real but unusual for unicorns
# Flag them rather than drop — they're valid data points
df["early_founder_flag"] = (df["year_founded"] < 1990).astype(int)

print(f"Early-founded companies (pre-1990): {df['early_founder_flag'].sum()}")
print(df[df["early_founder_flag"] == 1][["company", "year_founded"]].to_string(), "\n")

# ── 6. Feature engineering ────────────────────────────────────────────────────

# Age when it became a unicorn (years from founding to joining)
df["years_to_unicorn"] = df["year_joined"] - df["year_founded"]

# Clip extreme negative values (data entry errors where year_founded > year_joined)
n_neg = (df["years_to_unicorn"] < 0).sum()
if n_neg:
    print(f"Negative years_to_unicorn (clamped to 0): {n_neg} rows")
df["years_to_unicorn"] = df["years_to_unicorn"].clip(lower=0)

# Funding efficiency: valuation per $1B of funding raised
# Avoid divide-by-zero for companies with $0 funding recorded
df["funding_efficiency"] = np.where(
    df["funding_b"] > 0,
    df["valuation_b"] / df["funding_b"],
    np.nan,
)

# Binary target: high-value unicorn (>= $10B)
df["is_high_value"] = (df["valuation_b"] >= 10).astype(int)
print(f"High-value unicorns (>=$10B): {df['is_high_value'].sum()} "
      f"({df['is_high_value'].mean()*100:.1f}%)\n")

# Log-transformed valuation (for regression — raw valuation is right-skewed)
df["log_valuation"] = np.log1p(df["valuation_b"])

# Decade founded
df["decade_founded"] = (df["year_founded"] // 10 * 10).astype("Int64")

# ── 7. Handle missing values ──────────────────────────────────────────────────
print("Missing values before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0], "\n")

# City can be missing — fill with "Unknown"
df["city"] = df["city"].fillna("Unknown")

# For numeric columns used in ML, fill with median
for col in ["funding_b", "years_to_unicorn", "funding_efficiency"]:
    median_val = df[col].median()
    n_filled = df[col].isna().sum()
    if n_filled:
        df[col] = df[col].fillna(median_val)
        print(f"Filled {n_filled} missing '{col}' with median ({median_val:.2f})")

# ── 8. Strip & standardise string columns ────────────────────────────────────
for col in ["company", "industry", "city", "country", "continent"]:
    df[col] = df[col].str.strip()

# ── 9. Final checks ───────────────────────────────────────────────────────────
print("\n── Final dataset summary ──")
print(f"Shape: {df.shape}")
print(f"\nDtype overview:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nValuation stats ($B):\n{df['valuation_b'].describe().round(2)}")
print(f"\nYears to unicorn stats:\n{df['years_to_unicorn'].describe().round(1)}")
print(f"\nIndustry distribution:\n{df['industry'].value_counts()}")

# ── 10. Save ──────────────────────────────────────────────────────────────────
df.to_csv("data/unicorn_companies_clean.csv", index=False)
print("\nSaved → data/unicorn_companies_clean.csv")