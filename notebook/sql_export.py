"""
Phase 2b — Export to SQLite

Input:  data/unicorn_companies_clean.csv
Output: sql/unicorn.db  (SQLite database with two tables)

Run:  python3 notebook/sql_export.py| tee report/sqlReport.txt
"""

import pandas as pd
import sqlite3
import os

# ── 1. Load cleaned data ──────────────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies_clean.csv", parse_dates=["date_joined"])
print(f"Loaded {len(df):,} rows from data/unicorn_companies_clean.csv")

# ── 2. Connect (creates file if it doesn't exist) ─────────────────────────────
db_path = "sql/unicorn.db"
if os.path.exists(db_path):
    os.remove(db_path)          # fresh build each run
    print(f"Removed existing {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
print(f"Connected to {db_path}\n")

# ── 3. Main table ─────────────────────────────────────────────────────────────
df.to_sql("unicorns", conn, if_exists="replace", index=False)
print("Created table: unicorns")

# Verify
cursor.execute("SELECT COUNT(*) FROM unicorns")
print(f"  Rows inserted: {cursor.fetchone()[0]:,}")

# ── 4. Pre-aggregated industry summary table ──────────────────────────────────
industry_summary = (
    df.groupby("industry")
    .agg(
        company_count     = ("company",          "count"),
        avg_valuation_b   = ("valuation_b",      "mean"),
        median_valuation_b= ("valuation_b",      "median"),
        total_funding_b   = ("funding_b",        "sum"),
        avg_years_to_uni  = ("years_to_unicorn", "mean"),
        high_value_count  = ("is_high_value",    "sum"),
    )
    .reset_index()
    .round(2)
)

industry_summary.to_sql("industry_summary", conn, if_exists="replace", index=False)
print("\nCreated table: industry_summary")
cursor.execute("SELECT COUNT(*) FROM industry_summary")
print(f"  Rows inserted: {cursor.fetchone()[0]}")

# ── 5. Spot-check with some analytical queries ────────────────────────────────
print("\n── Sample queries ──\n")

# Top 10 most valuable unicorns
q1 = """
SELECT company, country, industry, valuation_b
FROM unicorns
ORDER BY valuation_b DESC
LIMIT 10
"""
print("Top 10 most valuable unicorns:")
print(pd.read_sql(q1, conn).to_string(index=False), "\n")

# Countries with most unicorns
q2 = """
SELECT country, COUNT(*) AS unicorn_count,
       ROUND(AVG(valuation_b), 2) AS avg_valuation_b
FROM unicorns
GROUP BY country
ORDER BY unicorn_count DESC
LIMIT 8
"""
print("Top 8 countries by unicorn count:")
print(pd.read_sql(q2, conn).to_string(index=False), "\n")

# Industry funding efficiency
q3 = """
SELECT industry,
       COUNT(*) AS companies,
       ROUND(AVG(funding_efficiency), 2) AS avg_funding_efficiency,
       ROUND(AVG(years_to_unicorn), 1) AS avg_years_to_unicorn
FROM unicorns
WHERE funding_efficiency IS NOT NULL
GROUP BY industry
ORDER BY avg_funding_efficiency DESC
"""
print("Industry funding efficiency (valuation per $1B raised):")
print(pd.read_sql(q3, conn).to_string(index=False), "\n")

# High-value unicorns by continent
q4 = """
SELECT continent,
       SUM(is_high_value) AS high_value_count,
       COUNT(*) AS total,
       ROUND(100.0 * SUM(is_high_value) / COUNT(*), 1) AS pct_high_value
FROM unicorns
GROUP BY continent
ORDER BY high_value_count DESC
"""
print("High-value unicorns (>=$10B) by continent:")
print(pd.read_sql(q4, conn).to_string(index=False))

# ── 6. Create indexes for query performance ───────────────────────────────────
cursor.execute("CREATE INDEX idx_industry  ON unicorns (industry)")
cursor.execute("CREATE INDEX idx_country   ON unicorns (country)")
cursor.execute("CREATE INDEX idx_continent ON unicorns (continent)")
cursor.execute("CREATE INDEX idx_year_joined ON unicorns (year_joined)")
conn.commit()
print("\nIndexes created on: industry, country, continent, year_joined")

conn.close()
print(f"\nSaved → {db_path}")
print("Tables: unicorns, industry_summary")

