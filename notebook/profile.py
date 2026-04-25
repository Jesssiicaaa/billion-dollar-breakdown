"""
Phase 1, Step 2 — Profile the raw data distributions

Run: python3 notebook/profile.py | tee report/profileReport.txt
"""

import pandas as pd
import numpy as np

CSV_PATH = "data/unicorn_companies.csv"

SEP  = "=" * 65
SEP2 = "-" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def hbar(value, max_value, width=25):
    filled = int(value / max_value * width)
    return "█" * filled + "░" * (width - filled)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["Date Joined"] = pd.to_datetime(df["Date Joined"], dayfirst=True, errors="coerce")
df["Year Joined"] = df["Date Joined"].dt.year

n = len(df)
print(f"\nProfiling {n:,} unicorn companies from raw dataset.\n")

# ── 1. Valuation distribution ─────────────────────────────────────────────────
section("1. VALUATION DISTRIBUTION ($B)")
bins   = [0, 1, 2, 5, 10, 25, 50, 100, 200]
labels = ["$1B", "$1–2B", "$2–5B", "$5–10B", "$10–25B", "$25–50B", "$50–100B", "$100B+"]
df["val_bin"] = pd.cut(df["Valuation ($Billions)"], bins=bins, labels=labels)
bin_counts = df["val_bin"].value_counts().sort_index()
max_c = bin_counts.max()
print(f"  {'Range':<14} {'Count':>6}  {'%':>5}  {'Bar'}")
print(f"  {SEP2}")
for label, cnt in bin_counts.items():
    pct = cnt / n * 100
    print(f"  {str(label):<14} {cnt:>6,}  {pct:>4.1f}%  {hbar(cnt, max_c)}")
print(f"\n  Mean  : ${df['Valuation ($Billions)'].mean():.2f}B")
print(f"  Median: ${df['Valuation ($Billions)'].median():.2f}B")
print(f"  Max   : ${df['Valuation ($Billions)'].max():.0f}B  "
      f"({df.loc[df['Valuation ($Billions)'].idxmax(), 'Company']})")

# ── 2. Industry distribution ──────────────────────────────────────────────────
section("2. INDUSTRY DISTRIBUTION")
ind = df["Industry"].value_counts()
max_c = ind.max()
print(f"  {'Industry':<40} {'Count':>6}  {'%':>5}  {'Bar'}")
print(f"  {SEP2}")
for name, cnt in ind.items():
    pct = cnt / n * 100
    label = str(name)[:38]
    print(f"  {label:<40} {cnt:>6,}  {pct:>4.1f}%  {hbar(cnt, max_c, 18)}")

# ── 3. Country distribution ───────────────────────────────────────────────────
section("3. COUNTRY DISTRIBUTION (top 15)")
ctry = df["Country"].value_counts().head(15)
max_c = ctry.max()
print(f"  {'Country':<25} {'Count':>6}  {'%':>5}  {'Bar'}")
print(f"  {SEP2}")
for name, cnt in ctry.items():
    pct = cnt / n * 100
    print(f"  {name:<25} {cnt:>6,}  {pct:>4.1f}%  {hbar(cnt, max_c)}")
remaining = n - ctry.sum()
print(f"\n  Remaining {df['Country'].nunique() - 15} countries: {remaining} companies")

# ── 4. Continent distribution ─────────────────────────────────────────────────
section("4. CONTINENT DISTRIBUTION")
cont = df["Continent"].value_counts()
max_c = cont.max()
print(f"  {'Continent':<20} {'Count':>6}  {'%':>5}  {'Bar'}")
print(f"  {SEP2}")
for name, cnt in cont.items():
    pct = cnt / n * 100
    print(f"  {name:<20} {cnt:>6,}  {pct:>4.1f}%  {hbar(cnt, max_c)}")

# ── 5. Founding year distribution ─────────────────────────────────────────────
section("5. FOUNDING YEAR DISTRIBUTION (by decade)")
df["Decade"] = (df["Year Founded"] // 10 * 10).astype("Int64")
decade = df["Decade"].value_counts().sort_index()
max_c = decade.max()
print(f"  {'Decade':<10} {'Count':>6}  {'%':>5}  {'Bar'}")
print(f"  {SEP2}")
for dec, cnt in decade.items():
    pct = cnt / n * 100
    print(f"  {str(dec)+'s':<10} {cnt:>6,}  {pct:>4.1f}%  {hbar(cnt, max_c)}")

# ── 6. Year joined (became unicorn) ──────────────────────────────────────────
section("6. YEAR JOINED (became unicorn)")
yr_joined = df["Year Joined"].dropna().astype(int).value_counts().sort_index()
max_c = yr_joined.max()
print(f"  {'Year':<8} {'Count':>6}  {'Bar'}")
print(f"  {SEP2}")
for yr, cnt in yr_joined.items():
    print(f"  {yr:<8} {cnt:>6,}  {hbar(cnt, max_c, 30)}")

# ── 7. Funding distribution ───────────────────────────────────────────────────
section("7. FUNDING DISTRIBUTION ($B)")
fund = df["Funding ($ Billions)"].dropna()
fbins   = [0, 0.5, 1, 2, 5, 10, 20, 50]
flabels = ["<$0.5B", "$0.5–1B", "$1–2B", "$2–5B", "$5–10B", "$10–20B", "$20B+"]
df["fund_bin"] = pd.cut(fund, bins=fbins, labels=flabels)
fbin_counts = df["fund_bin"].value_counts().sort_index()
max_c = fbin_counts.max()
print(f"  {'Range':<14} {'Count':>6}  {'%':>5}  {'Bar'}")
print(f"  {SEP2}")
for label, cnt in fbin_counts.items():
    pct = cnt / fund.notna().sum() * 100
    print(f"  {str(label):<14} {cnt:>6,}  {pct:>4.1f}%  {hbar(cnt, max_c)}")
print(f"\n  Mean  : ${fund.mean():.2f}B")
print(f"  Median: ${fund.median():.2f}B")
print(f"  Max   : ${fund.max():.0f}B  "
      f"({df.loc[df['Funding ($ Billions)'].idxmax(), 'Company']})")

# ── 8. Valuation by continent ─────────────────────────────────────────────────
section("8. AVG VALUATION BY CONTINENT ($B)")
cont_val = (
    df.groupby("Continent")["Valuation ($Billions)"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
    .round(2)
)
max_c = cont_val["mean"].max()
print(f"  {'Continent':<20} {'Mean':>7}  {'Median':>7}  {'Count':>6}  {'Bar'}")
print(f"  {SEP2}")
for cont, row in cont_val.iterrows():
    print(f"  {cont:<20} ${row['mean']:>5.2f}B  ${row['median']:>5.2f}B"
          f"  {int(row['count']):>6,}  {hbar(row['mean'], max_c, 18)}")

# ── 9. Top companies per industry ─────────────────────────────────────────────
section("9. HIGHEST-VALUED COMPANY PER INDUSTRY")
top_per_ind = (
    df.loc[df.groupby("Industry")["Valuation ($Billions)"].idxmax(),
           ["Industry", "Company", "Valuation ($Billions)", "Country"]]
    .sort_values("Valuation ($Billions)", ascending=False)
)
print(top_per_ind.to_string(index=False))

# ── 10. Summary ───────────────────────────────────────────────────────────────
section("10. QUICK FACTS")
print(f"  Total unicorn companies       : {n:,}")
print(f"  Total combined valuation      : ${df['Valuation ($Billions)'].sum():,.0f}B")
print(f"  Total combined funding        : ${df['Funding ($ Billions)'].sum():,.0f}B")
print(f"  Countries represented         : {df['Country'].nunique()}")
print(f"  Industries represented        : {df['Industry'].nunique()}")
print(f"  Median years to become unicorn: ~{(df['Year Joined'] - df['Year Founded']).median():.0f} yrs (approx)")
print(f"  Most common founding decade   : {df['Decade'].mode()[0]}s")
print(f"  Peak year for new unicorns    : {yr_joined.idxmax()}"
      f"  ({yr_joined.max()} companies)")

print(f"\n{SEP}")
print("  Profiling complete. No data was modified.")
print(f"  Next step → phase1_step3_charts.py")
print(SEP)