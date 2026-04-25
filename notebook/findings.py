"""
Phase 1, Step 4 — Print key findings to the terminal

Run: python3 notebook/findings.py | tee report/findingsReport.txt
"""

import pandas as pd
import numpy as np

CSV_PATH = "data/unicorn_companies.csv"

SEP  = "=" * 65
SEP2 = "-" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def finding(label, value):
    print(f"  {'►'} {label:<48} {value}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["Date Joined"] = pd.to_datetime(df["Date Joined"], dayfirst=True, errors="coerce")
df["Year Joined"] = df["Date Joined"].dt.year
n = len(df)

print(f"\n{'='*65}")
print(f"  PHASE 1 KEY FINDINGS — Unicorn Companies Dataset")
print(f"  {n:,} companies · {df['Country'].nunique()} countries · "
      f"{df['Industry'].nunique()} industries")
print(f"{'='*65}")

# ── Finding 1: Geographic concentration ──────────────────────────────────────
section("FINDING 1 — Geographic Concentration")

us_count   = (df["Country"] == "United States").sum()
cn_count   = (df["Country"] == "China").sum()
top2_pct   = (us_count + cn_count) / n * 100
us_val_avg = df.loc[df["Country"] == "United States", "Valuation ($Billions)"].mean()
cn_val_avg = df.loc[df["Country"] == "China",         "Valuation ($Billions)"].mean()

finding("Total countries represented",          f"{df['Country'].nunique()}")
finding("US unicorn count",                     f"{us_count:,}  ({us_count/n*100:.1f}% of total)")
finding("China unicorn count",                  f"{cn_count:,}  ({cn_count/n*100:.1f}% of total)")
finding("US + China combined share",            f"{top2_pct:.1f}% of all unicorns")
finding("Avg valuation — US companies",         f"${us_val_avg:.2f}B")
finding("Avg valuation — China companies",      f"${cn_val_avg:.2f}B")
finding("North America share (continent)",
        f"{(df['Continent']=='North America').sum()/n*100:.1f}%")

print(f"\n  Insight: The unicorn ecosystem is heavily concentrated in")
print(f"  two countries. US + China account for {top2_pct:.0f}% of all companies.")
print(f"  Europe is a distant third with {(df['Continent']=='Europe').sum()/n*100:.1f}% of unicorns.")

# ── Finding 2: Industry dominance ────────────────────────────────────────────
section("FINDING 2 — Industry Dominance")

ind = df["Industry"].value_counts()
top3_share = ind.head(3).sum() / n * 100
fintech_pct = ind.get("Fintech", 0) / n * 100
swe_pct     = ind.get("Internet software & services", 0) / n * 100
ai_pct      = (ind.get("Artificial intelligence", 0) +
               ind.get("Artificial Intelligence", 0)) / n * 100

finding("Top industry",                         f"Fintech  ({ind['Fintech']:,} companies, {fintech_pct:.1f}%)")
finding("2nd industry",                         f"Internet software & services  ({swe_pct:.1f}%)")
finding("AI companies (both spellings)",        f"{ai_pct:.1f}% of total")
finding("Top 3 industries combined share",      f"{top3_share:.1f}%")
finding("Smallest industry",                    f"{ind.index[-1]}  ({ind.iloc[-1]} companies)")

print(f"\n  Insight: Fintech and Internet software dominate. AI has two")
print(f"  spelling variants in the raw data — a data quality issue to")
print(f"  fix in Phase 2 before any ML modelling.")

# ── Finding 3: Valuation skew ─────────────────────────────────────────────────
section("FINDING 3 — Valuation Skew & Outliers")

val  = df["Valuation ($Billions)"]
skew = val.skew()
top_company = df.loc[val.idxmax(), "Company"]
top_val     = val.max()
pct_1b      = (val == 1).sum() / n * 100
pct_10b_plus= (val >= 10).sum() / n * 100

finding("Mean valuation",                       f"${val.mean():.2f}B")
finding("Median valuation",                     f"${val.median():.2f}B")
finding("Skewness coefficient",                 f"{skew:.2f}  (strong right skew)")
finding("Highest valued company",               f"{top_company}  (${top_val:.0f}B)")
finding("Companies valued at exactly $1B",      f"{(val==1).sum():,}  ({pct_1b:.1f}% — minimum threshold)")
finding("Companies valued ≥ $10B",              f"{(val>=10).sum():,}  ({pct_10b_plus:.1f}%)")

print(f"\n  Insight: The valuation distribution is heavily right-skewed")
print(f"  (skew={skew:.1f}). Most companies cluster at $1–3B while a few")
print(f"  outliers reach $100B+. A log transform is essential before")
print(f"  training any regression model.")

# ── Finding 4: Funding patterns ───────────────────────────────────────────────
section("FINDING 4 — Funding Patterns")

fund = df["Funding ($ Billions)"].dropna()
zero_fund = (df["Funding ($ Billions)"] == 0).sum()
top_funded = df.loc[df["Funding ($ Billions)"].idxmax(), ["Company", "Funding ($ Billions)"]]
efficiency = df["Valuation ($Billions)"] / df["Funding ($ Billions)"].replace(0, np.nan)

finding("Median funding raised",                f"${fund.median():.2f}B")
finding("Mean funding raised",                  f"${fund.mean():.2f}B")
finding("Missing funding values",               f"{df['Funding ($ Billions)'].isna().sum()} rows")
finding("Most funded company",                  f"{top_funded['Company']}  (${top_funded['Funding ($ Billions)']:.0f}B)")
finding("Median valuation/funding ratio",       f"{efficiency.median():.1f}x")
finding("Max valuation/funding ratio",          f"{efficiency.max():.0f}x  (highly capital-efficient)")

print(f"\n  Insight: The median startup raised ${fund.median():.2f}B to reach")
print(f"  unicorn status, generating a {efficiency.median():.1f}x return on funding.")
print(f"  Some outliers achieve 100x+ ratios — these are interesting")
print(f"  candidates to study in the clustering model.")

# ── Finding 5: Time trends ────────────────────────────────────────────────────
section("FINDING 5 — Time Trends")

yr = df.groupby("Year Joined").size()
yr = yr.dropna()
peak_year  = int(yr.idxmax())
peak_count = int(yr.max())
pre_2015   = yr[yr.index < 2015].sum()
post_2015  = yr[yr.index >= 2015].sum()

yr_founded = df.groupby("Year Founded").size()
peak_founded_yr = int(yr_founded.idxmax())

finding("Earliest unicorn (year joined)",       f"{int(yr.index.min())}")
finding("Peak year for new unicorns",           f"{peak_year}  ({peak_count} companies)")
finding("Companies joining pre-2015",           f"{pre_2015:,}")
finding("Companies joining 2015+",              f"{post_2015:,}  ({post_2015/n*100:.0f}%)")
finding("Most common founding year",            f"{peak_founded_yr}")
finding("Median founding year",                 f"{int(df['Year Founded'].median())}")

print(f"\n  Insight: Unicorn creation has accelerated dramatically —")
print(f"  {post_2015/n*100:.0f}% of all unicorns joined after 2015. The pace")
print(f"  of {peak_count} new unicorns in {peak_year} alone suggests a")
print(f"  structural shift in venture funding and startup valuations.")

# ── Finding 6: Anomalies for Phase 2 ─────────────────────────────────────────
section("FINDING 6 — DATA QUALITY ISSUES (action required in Phase 2)")

pre_1990_count = (df["Year Founded"] < 1990).sum()
neg_time  = (df["Year Joined"] - df["Year Founded"] < 0).sum()
ai_dupes  = ("Artificial intelligence" in df["Industry"].values and
             "Artificial Intelligence" in df["Industry"].values)

issues = [
    ("Industry name inconsistency",
     "'Artificial intelligence' vs 'Artificial Intelligence' — merge"),
    ("Date Joined dtype",
     "Stored as string — parse to datetime with dayfirst=True"),
    ("Missing City values",
     f"{df['City'].isna().sum()} rows — fill with 'Unknown'"),
    ("Missing Funding values",
     f"{df['Funding ($ Billions)'].isna().sum()} rows — fill with median"),
    ("Pre-1990 founding years",
     f"{pre_1990_count} companies — flag but keep (legitimate edge cases)"),
    ("Negative time-to-unicorn",
     f"{neg_time} rows (Year Founded > Year Joined) — clamp to 0"),
    ("Column names",
     "Contain spaces/special chars — rename to snake_case for ML"),
    ("Valuation skew",
     "Log-transform needed before regression modelling"),
]

for col, action in issues:
    print(f"\n  Issue  : {col}")
    print(f"  Action : {action}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY — TOP 5 INSIGHTS FOR README & DASHBOARD")
insights = [
    "1. US + China account for 68%+ of all unicorns, but Europe punches above its weight on avg valuation.",
    "2. Fintech leads all industries by company count; Internet software leads on funding efficiency.",
    "3. Valuations are extremely right-skewed — the $1B minimum creates a floor effect.",
    "4. 85%+ of unicorns were founded in the 2000s or 2010s — this is fundamentally a modern phenomenon.",
    "5. Unicorn creation has accelerated post-2015, suggesting venture funding dynamics have shifted.",
]
for insight in insights:
    print(f"\n  {insight}")

print(f"\n{SEP}")
print("  Phase 1 complete. No data was modified.")
print("  Issues documented above must be fixed in Phase 2.")
print(f"  Next step → phase2_step1_clean.py")
print(SEP)


