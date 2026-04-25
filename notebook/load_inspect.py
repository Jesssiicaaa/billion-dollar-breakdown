"""
load_inspect.py
------------------
Phase 1, Step 1 — Load the raw CSV and print a full inspection report.
Run: python3 notebook/load_inspect.py | tee report/inspectionReport.txt
"""

import pandas as pd

CSV_PATH = "data/unicorn_companies.csv"

# ── Load ────────────────────────────────────────────────────────
# ── Formatting helpers ────────────────────────────────────────────────────────
SEP  = "=" * 65
SEP2 = "-" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── 2. Shape & memory ─────────────────────────────────────────────────────────
section("1. BASIC INFO")
print(f"  File     : {CSV_PATH}")
print(f"  Rows     : {df.shape[0]:,}")
print(f"  Columns  : {df.shape[1]}")
mem = df.memory_usage(deep=True).sum() / 1024
print(f"  Memory   : {mem:.1f} KB")

# ── 3. Columns, dtypes, sample values ────────────────────────────────────────
section("2. COLUMNS — DTYPES & SAMPLE VALUES")
print(f"  {'Column':<30} {'Dtype':<12} {'Sample Value'}")
print(f"  {SEP2}")
for col in df.columns:
    sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A"
    if len(sample) > 30:
        sample = sample[:27] + "..."
    print(f"  {col:<30} {str(df[col].dtype):<12} {sample}")

# ── 4. Missing values ─────────────────────────────────────────────────────────
section("3. MISSING VALUES")
nulls = df.isnull().sum()
if nulls.sum() == 0:
    print("  No missing values found.")
else:
    print(f"  {'Column':<30} {'Missing':>8}  {'%':>6}")
    print(f"  {SEP2}")
    for col, n in nulls[nulls > 0].items():
        pct = n / len(df) * 100
        print(f"  {col:<30} {n:>8,}  {pct:>5.1f}%")

# ── 5. Duplicate rows ─────────────────────────────────────────────────────────
section("4. DUPLICATE ROWS")
n_dupes = df.duplicated().sum()
print(f"  Exact duplicate rows: {n_dupes}")
if n_dupes > 0:
    print(df[df.duplicated(keep=False)])

# ── 6. Numeric column statistics ──────────────────────────────────────────────
section("5. NUMERIC COLUMNS — DESCRIPTIVE STATS")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print(df[numeric_cols].describe().round(2).to_string())

# ── 7. Categorical column value counts ───────────────────────────────────────
section("6. CATEGORICAL COLUMNS — VALUE COUNTS (top 10)")
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    n_unique = df[col].nunique()
    print(f"\n  [{col}]  —  {n_unique} unique values")
    print(f"  {SEP2}")
    counts = df[col].value_counts().head(10)
    for val, cnt in counts.items():
        bar = "█" * int(cnt / counts.max() * 20)
        pct = cnt / len(df) * 100
        label = str(val)[:35]
        print(f"  {label:<36} {cnt:>5,}  {pct:>5.1f}%  {bar}")

# ── 8. Date column format check ───────────────────────────────────────────────
section("7. DATE COLUMN — FORMAT CHECK")
date_col = "Date Joined"
sample_dates = df[date_col].dropna().head(10).tolist()
print(f"  Column   : '{date_col}'")
print(f"  Dtype    : {df[date_col].dtype}  (stored as string — needs parsing)")
print(f"\n  Sample values:")
for d in sample_dates:
    print(f"    {d}")

parsed = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
n_failed = parsed.isna().sum()
print(f"\n  Parse test (dayfirst=True): {len(df) - n_failed:,} succeeded, "
      f"{n_failed} failed")
if n_failed == 0:
    print(f"  Date range: {parsed.min().date()} → {parsed.max().date()}")

# ── 9. Year Founded check ─────────────────────────────────────────────────────
section("8. YEAR FOUNDED — OUTLIER CHECK")
yr = df["Year Founded"]
print(f"  Min: {yr.min()}   Max: {yr.max()}")
pre_1990 = df[yr < 1990][["Company", "Year Founded"]]
print(f"\n  Companies founded before 1990 ({len(pre_1990)} found):")
if len(pre_1990):
    print(pre_1990.to_string(index=False))
else:
    print("  None.")

# ── 10. Valuation skew check ──────────────────────────────────────────────────
section("9. VALUATION — SKEW & OUTLIER CHECK")
val = df["Valuation ($Billions)"]
print(f"  Mean   : ${val.mean():.2f}B")
print(f"  Median : ${val.median():.2f}B")
print(f"  Skew   : {val.skew():.2f}  (>1 = right-skewed, needs log transform for ML)")
print(f"\n  Top 10 most valuable:")
top10 = df.nlargest(10, "Valuation ($Billions)")[
    ["Company", "Valuation ($Billions)", "Industry", "Country"]
]
print(top10.to_string(index=False))

# ── 11. Summary of issues found ───────────────────────────────────────────────
section("10. DATA QUALITY ISSUES — SUMMARY FOR PHASE 2")
issues = [
    ("Date Joined",            "Stored as string — must be parsed to datetime"),
    ("City",                   f"{nulls.get('City', 0)} missing values — fill with 'Unknown'"),
    ("Funding ($ Billions)",   f"{nulls.get('Funding ($ Billions)', 0)} missing values — fill with median"),
    ("Year Founded",           f"{len(pre_1990)} pre-1990 outliers — flag but keep"),
    ("Valuation ($Billions)",  f"Skew = {val.skew():.2f} — log-transform before ML regression"),
    ("Column names",           "Contain spaces & special chars — rename to snake_case"),
    ("Industry",               "Check for near-duplicate categories (e.g. case differences)"),
]
print(f"  {'Column':<30} {'Issue'}")
print(f"  {SEP2}")
for col, issue in issues:
    print(f"  {col:<30} {issue}")

print(f"\n{SEP}")
print("  Inspection complete. No data was modified.")
print(f"  Next step → phase1_step2_profile.py")
print(SEP)