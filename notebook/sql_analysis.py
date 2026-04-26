"""
Phase 3, Step 1 — SQL Analysis
Unicorn Companies Project
------------------------------------------------
Input:  unicorn.db  (from phase2b_sql_export.py)
Output: sql/results/  folder with one CSV per query
        Prints all results to terminal

Run:  python3 notebook/sql_analysis.py

Queries:
  Q01 — Top 20 most valuable unicorns
  Q02 — Unicorn count and avg valuation by country (top 15)
  Q03 — Industry ranking by total valuation
  Q04 — Funding efficiency leaders (valuation per $1B raised)
  Q05 — Fastest companies to reach unicorn status
  Q06 — High-value unicorn rate by industry
  Q07 — Unicorn creation by year (growth trend)
  Q08 — Continent vs industry cross-tab (pivot)
  Q09 — Late bloomers — founded before 2000, joined after 2015
  Q10 — Decade-over-decade founding patterns
"""

import sqlite3
import pandas as pd
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
DB_PATH     = "sql/unicorn.db"
OUTPUT_DIR  = "sql/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
print(f"Connected to {DB_PATH}\n")
print("=" * 65)

# ── Query runner ──────────────────────────────────────────────────────────────
results = {}

def run_query(label, qid, sql, note=""):
    print(f"\n{'─' * 65}")
    print(f"  {qid} — {label}")
    if note:
        print(f"  Note: {note}")
    print(f"{'─' * 65}")
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))
    filename = f"{OUTPUT_DIR}/{qid}_{label.lower().replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    results[qid] = df
    return df

# ── Q01: Top 20 most valuable unicorns ───────────────────────────────────────
run_query("top_20_by_valuation", "Q01", """
    SELECT
        company,
        valuation_b,
        industry,
        country,
        year_founded,
        year_joined,
        years_to_unicorn
    FROM unicorns
    ORDER BY valuation_b DESC
    LIMIT 20
""")

# ── Q02: Country ranking ──────────────────────────────────────────────────────
run_query("country_ranking", "Q02", """
    SELECT
        country,
        continent,
        COUNT(*)                          AS unicorn_count,
        ROUND(AVG(valuation_b), 2)        AS avg_valuation_b,
        ROUND(SUM(valuation_b), 0)        AS total_valuation_b,
        ROUND(AVG(years_to_unicorn), 1)   AS avg_years_to_unicorn,
        SUM(is_high_value)                AS high_value_count
    FROM unicorns
    GROUP BY country, continent
    ORDER BY unicorn_count DESC
    LIMIT 15
""")

# ── Q03: Industry ranking by total valuation ──────────────────────────────────
run_query("industry_total_valuation", "Q03", """
    SELECT
        industry,
        COUNT(*)                            AS companies,
        ROUND(SUM(valuation_b), 1)          AS total_valuation_b,
        ROUND(AVG(valuation_b), 2)          AS avg_valuation_b,
        ROUND(AVG(funding_b), 2)            AS avg_funding_b,
        ROUND(AVG(years_to_unicorn), 1)     AS avg_years_to_unicorn
    FROM unicorns
    GROUP BY industry
    ORDER BY total_valuation_b DESC
""")

# ── Q04: Funding efficiency leaders ──────────────────────────────────────────
run_query("funding_efficiency_leaders", "Q04", """
    SELECT
        company,
        industry,
        country,
        valuation_b,
        funding_b,
        ROUND(funding_efficiency, 1)    AS valuation_per_b_raised,
        years_to_unicorn
    FROM unicorns
    WHERE funding_b > 0
      AND funding_efficiency IS NOT NULL
    ORDER BY funding_efficiency DESC
    LIMIT 20
""", note="valuation_per_b_raised = valuation / funding raised")

# ── Q05: Fastest to unicorn status ───────────────────────────────────────────
run_query("fastest_to_unicorn", "Q05", """
    SELECT
        company,
        industry,
        country,
        year_founded,
        year_joined,
        years_to_unicorn,
        valuation_b
    FROM unicorns
    WHERE years_to_unicorn >= 0
    ORDER BY years_to_unicorn ASC
    LIMIT 20
""")

# ── Q06: High-value rate by industry ─────────────────────────────────────────
run_query("high_value_rate_by_industry", "Q06", """
    SELECT
        industry,
        COUNT(*)                                        AS total_companies,
        SUM(is_high_value)                              AS high_value_count,
        ROUND(100.0 * SUM(is_high_value) / COUNT(*), 1) AS high_value_pct,
        ROUND(AVG(valuation_b), 2)                      AS avg_valuation_b,
        MAX(valuation_b)                                AS max_valuation_b
    FROM unicorns
    GROUP BY industry
    ORDER BY high_value_pct DESC
""")

# ── Q07: Unicorn creation by year ─────────────────────────────────────────────
run_query("unicorn_creation_by_year", "Q07", """
    SELECT
        year_joined,
        COUNT(*)                            AS new_unicorns,
        ROUND(AVG(valuation_b), 2)          AS avg_valuation_b,
        ROUND(SUM(valuation_b), 1)          AS total_valuation_b,
        ROUND(AVG(years_to_unicorn), 1)     AS avg_years_to_unicorn,
        SUM(is_high_value)                  AS high_value_count
    FROM unicorns
    WHERE year_joined IS NOT NULL
    GROUP BY year_joined
    ORDER BY year_joined ASC
""")

# ── Q08: Continent × industry cross-tab ──────────────────────────────────────
run_query("continent_industry_crosstab", "Q08", """
    SELECT
        continent,
        industry,
        COUNT(*)                        AS companies,
        ROUND(AVG(valuation_b), 2)      AS avg_valuation_b
    FROM unicorns
    GROUP BY continent, industry
    ORDER BY continent, companies DESC
""")

# ── Q09: Late bloomers ────────────────────────────────────────────────────────
run_query("late_bloomers", "Q09", """
    SELECT
        company,
        industry,
        country,
        year_founded,
        year_joined,
        years_to_unicorn,
        valuation_b,
        funding_b
    FROM unicorns
    WHERE year_founded < 2000
      AND year_joined  > 2015
    ORDER BY years_to_unicorn DESC
""", note="Founded before 2000 but only reached unicorn status after 2015")

# ── Q10: Decade-over-decade founding patterns ─────────────────────────────────
run_query("decade_founding_patterns", "Q10", """
    SELECT
        decade_founded,
        COUNT(*)                            AS companies_founded,
        ROUND(AVG(valuation_b), 2)          AS avg_valuation_b,
        ROUND(AVG(years_to_unicorn), 1)     AS avg_years_to_unicorn,
        ROUND(AVG(funding_b), 2)            AS avg_funding_b,
        SUM(is_high_value)                  AS high_value_count
    FROM unicorns
    WHERE decade_founded IS NOT NULL
    GROUP BY decade_founded
    ORDER BY decade_founded ASC
""")

# ── Summary ───────────────────────────────────────────────────────────────────
conn.close()
print("\n" + "=" * 65)
print(f"All 10 queries complete.")
print(f"CSVs saved to: {OUTPUT_DIR}/")
print("=" * 65)