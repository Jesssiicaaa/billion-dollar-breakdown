"""
clean.py
-----------
Phase 1, Step 2 — Clean the raw data and save unicorn_companies_clean.csv.
Run: python3 notebook/clean.py | tee report/cleanReport.txt
Requires: data/unicorn_companies.csv
Produces: data/unicorn_companies_clean.csv
"""

import pandas as pd

CSV_IN  = "data/unicorn_companies.csv"
CSV_OUT = "data/unicorn_companies_clean.csv"

df = pd.read_csv(CSV_IN)
print(f"Loaded {len(df):,} rows.\n")
"""reads the CSV file specified by CSV_IN into a pandas DataFrame and prints the number of rows loaded. The len(df) expression retrieves
the number of rows in the DataFrame, and the :,. format specifier formats it with commas for thousands separators. The \n at the end of the
print statement adds a newline """

# ── Step 1: Parse Date Joined ────────────────────────────────────
# Format in the file is DD/MM/YYYY
df["Date Joined"] = pd.to_datetime(df["Date Joined"], dayfirst=True)
"""dates in the "Date Joined" column are parsed into datetime objects using the pd.to_datetime() function.
The dayfirst=True argument indicates that the date format in the CSV file is day/month/year (DD/MM/YYYY)."""
df["Year Joined"]  = df["Date Joined"].dt.year
"""The dt.year attribute  extracts the year from the "Date Joined" column and creates a new column called "Year Joined"."""
df["Month Joined"] = df["Date Joined"].dt.month
"""The dt.month attribute  extracts the month from the "Date Joined" column and creates a new column called "Month Joined"."""
print(f"[1] Date range: {df['Date Joined'].min().date()} → {df['Date Joined'].max().date()}")
"""prints the minimum and maximum dates in the "Date Joined" column, formatted as date objects. The min() and max() methods are used to find
the earliest and latest dates, respectively, and the .date() method converts the datetime objects to date objects for cleaner output."""

# ── Step 2: Standardise Industry labels ─────────────────────────
# 'Artificial intelligence' and 'Artificial Intelligence' are duplicates
before = df["Industry"].nunique()
df["Industry"] = (
    df["Industry"]
    .str.strip()
    .str.replace(r"^Artificial [Ii]ntelligence$", "Artificial Intelligence", regex=True)
)
after = df["Industry"].nunique()
print(f"[2] Industry labels: {before} → {after} unique values after standardising")

# ── Step 3: Handle missing values ───────────────────────────────
# City — 16 nulls → 'Unknown' (small enough to keep)
df["City"] = df["City"].fillna("Unknown")

# Funding — 12 nulls → fill with industry median (robust to skew)
df["Funding ($ Billions)"] = df.groupby("Industry")["Funding ($ Billions)"].transform(
    lambda x: x.fillna(x.median())
)
remaining = df.isnull().sum().sum()
print(f"[3] Nulls remaining after fill: {remaining}")

# ── Step 4: Engineer new columns ────────────────────────────────
df["Years to Unicorn"]  = df["Year Joined"] - df["Year Founded"]
df["Val/Funding Ratio"] = (df["Valuation ($Billions)"] / df["Funding ($ Billions)"]).round(2)

# Drop rows where Years to Unicorn is negative (data entry errors)
invalid = df[df["Years to Unicorn"] < 0]
if len(invalid):
    print(f"[4] Dropping {len(invalid)} rows with negative Years to Unicorn:")
    print(invalid[["Company", "Year Founded", "Year Joined"]].to_string())
    df = df[df["Years to Unicorn"] >= 0].copy()
else:
    print("[4] No negative Years to Unicorn — all rows kept")

# ── Save ─────────────────────────────────────────────────────────
df.to_csv(CSV_OUT, index=False)
print(f"\nSaved → {CSV_OUT}  ({len(df):,} rows × {df.shape[1]} columns)")
print("\nFinal columns:")
for col in df.columns:
    print(f"  {col:<30} {df[col].dtype}")

