"""
Phase 3, Step 4 — Key Findings Report
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: report/findingsReport.txt  (also printed to terminal)

Run:  python3 notebook/findings_report.py

Generates a structured written findings summary — the narrative
that goes directly into your GitHub README and Tableau annotations.
Covers 6 themes:
  1. Market overview
  2. Geography
  3. Industry
  4. Speed to unicorn
  5. Funding efficiency
  6. Growth trends
"""

import pandas as pd
import numpy as np
import textwrap
import os

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies_clean.csv", parse_dates=["date_joined"])
df["industry"] = df["industry"].replace(
    "Artificial Intelligence", "Artificial intelligence"
)

# ── Pre-compute all stats ──────────────────────────────────────────────────────
total          = len(df)
n_countries    = df["country"].nunique()
n_industries   = df["industry"].nunique()
total_val      = df["valuation_b"].sum()
avg_val        = df["valuation_b"].mean()
median_val     = df["valuation_b"].median()
max_company    = df.loc[df["valuation_b"].idxmax(), "company"]
max_val        = df["valuation_b"].max()
high_value_n   = df["is_high_value"].sum()
high_value_pct = df["is_high_value"].mean() * 100

# Geography
top3_countries = df["country"].value_counts().head(3)
us_pct         = top3_countries.iloc[0] / total * 100
cn_pct         = top3_countries.iloc[1] / total * 100
in_pct         = top3_countries.iloc[2] / total * 100
na_pct         = df[df["continent"] == "North America"].shape[0] / total * 100
asia_pct       = df[df["continent"] == "Asia"].shape[0] / total * 100
eu_pct         = df[df["continent"] == "Europe"].shape[0] / total * 100

# Industry
top_industry       = df["industry"].value_counts().idxmax()
top_industry_n     = df["industry"].value_counts().max()
top_industry_pct   = top_industry_n / total * 100
second_industry    = df["industry"].value_counts().index[1]
second_industry_n  = df["industry"].value_counts().iloc[1]

industry_val = df.groupby("industry")["valuation_b"].mean().sort_values(ascending=False)
highest_avg_val_industry = industry_val.idxmax()
highest_avg_val          = industry_val.max()

# Speed
avg_yrs    = df["years_to_unicorn"].mean()
median_yrs = df["years_to_unicorn"].median()
fastest_n  = (df["years_to_unicorn"] <= 2).sum()
fastest_pct= fastest_n / total * 100
slowest    = df.loc[df["years_to_unicorn"].idxmax(), ["company", "years_to_unicorn"]]

fastest_industry = (df.groupby("industry")["years_to_unicorn"]
                    .median().idxmin())
fastest_ind_yrs  = (df.groupby("industry")["years_to_unicorn"]
                    .median().min())
slowest_industry = (df.groupby("industry")["years_to_unicorn"]
                    .median().idxmax())
slowest_ind_yrs  = (df.groupby("industry")["years_to_unicorn"]
                    .median().max())

# Funding efficiency
top_eff = (df[df["funding_b"] > 0]
           .nlargest(5, "funding_efficiency")
           [["company", "valuation_b", "funding_b", "funding_efficiency"]])
most_eff_industry = (df[df["funding_b"] > 0]
                     .groupby("industry")["funding_efficiency"]
                     .mean().idxmax())
most_eff_val      = (df[df["funding_b"] > 0]
                     .groupby("industry")["funding_efficiency"]
                     .mean().max())

# Trends
yearly = (df.groupby("year_joined")["company"]
          .count().dropna().sort_index())
peak_year  = int(yearly.idxmax())
peak_count = int(yearly.max())
first_year = int(yearly.index.min())
last_year  = int(yearly.index.max())
decade_growth = (df[df["year_joined"] >= 2015].shape[0] /
                 df[df["year_joined"] < 2015].shape[0])

# ── Format helpers ─────────────────────────────────────────────────────────────
W = 65   # line width

def section(title):
    return f"\n{'═' * W}\n  {title}\n{'═' * W}"

def bullet(text, indent=2):
    prefix = " " * indent + "• "
    return textwrap.fill(text, width=W,
                         initial_indent=prefix,
                         subsequent_indent=" " * (indent + 2))

def stat(label, value):
    dots = "." * max(1, W - len(label) - len(str(value)) - 4)
    return f"  {label} {dots} {value}"

# ── Build report ──────────────────────────────────────────────────────────────
lines = []
lines.append("=" * W)
lines.append("  UNICORN COMPANIES — KEY FINDINGS REPORT")
lines.append("  Global Unicorn Landscape: What Makes a $1B+ Company?")
lines.append("=" * W)

# ── 1. Market Overview
lines.append(section("1. Market Overview"))
lines.append(stat("Total unicorn companies",    f"{total:,}"))
lines.append(stat("Countries represented",      f"{n_countries}"))
lines.append(stat("Industries covered",         f"{n_industries}"))
lines.append(stat("Combined valuation",         f"${total_val:,.0f}B"))
lines.append(stat("Average valuation",          f"${avg_val:.1f}B"))
lines.append(stat("Median valuation",           f"${median_val:.1f}B"))
lines.append(stat("Most valuable company",      f"{max_company} (${max_val:.0f}B)"))
lines.append(stat("High-value unicorns (≥$10B)",f"{high_value_n} ({high_value_pct:.1f}%)"))
lines.append("")
lines.append(bullet(
    f"The median valuation of ${median_val:.0f}B versus the mean of "
    f"${avg_val:.1f}B reveals a strongly right-skewed market — a small "
    f"number of mega-unicorns pull the average well above where most "
    f"companies actually sit."
))
lines.append(bullet(
    f"Only {high_value_pct:.1f}% of unicorns have crossed the $10B mark, "
    f"confirming that reaching $1B status and reaching $10B+ are "
    f"fundamentally different achievements."
))

# ── 2. Geography
lines.append(section("2. Geography"))
lines.append(stat("United States share",  f"{us_pct:.1f}% ({top3_countries.iloc[0]} companies)"))
lines.append(stat("China share",          f"{cn_pct:.1f}% ({top3_countries.iloc[1]} companies)"))
lines.append(stat("India share",          f"{in_pct:.1f}% ({top3_countries.iloc[2]} companies)"))
lines.append(stat("North America total",  f"{na_pct:.1f}% of all unicorns"))
lines.append(stat("Asia total",           f"{asia_pct:.1f}% of all unicorns"))
lines.append(stat("Europe total",         f"{eu_pct:.1f}% of all unicorns"))
lines.append("")
lines.append(bullet(
    f"The US ({us_pct:.0f}%) and China ({cn_pct:.0f}%) together account for "
    f"more than {us_pct + cn_pct:.0f}% of all unicorns globally, but India "
    f"at {in_pct:.0f}% is rapidly emerging as a third major hub, "
    f"particularly in Fintech and SaaS."
))
lines.append(bullet(
    f"Europe ({eu_pct:.0f}%) and the rest of the world remain significantly "
    f"underrepresented relative to GDP, suggesting geographic opportunity "
    f"gaps in startup ecosystems."
))

# ── 3. Industry
lines.append(section("3. Industry"))
lines.append(stat("Largest industry by count",
                  f"{top_industry} ({top_industry_n} companies, {top_industry_pct:.0f}%)"))
lines.append(stat("Second largest",
                  f"{second_industry} ({second_industry_n} companies)"))
lines.append(stat("Highest avg valuation industry",
                  f"{highest_avg_val_industry} (${highest_avg_val:.1f}B avg)"))
lines.append("")
lines.append(bullet(
    f"{top_industry} leads with {top_industry_n} unicorns ({top_industry_pct:.0f}% "
    f"of the dataset), closely followed by {second_industry}. These two "
    f"industries alone represent over a third of all unicorns globally."
))
lines.append(bullet(
    f"{highest_avg_val_industry} produces the highest average valuation "
    f"(${highest_avg_val:.1f}B), suggesting that while it may not generate "
    f"the most unicorns by count, the ones it does produce tend to be "
    f"disproportionately valuable."
))

# ── 4. Speed to Unicorn
lines.append(section("4. Speed to Unicorn Status"))
lines.append(stat("Average years to unicorn",  f"{avg_yrs:.1f} years"))
lines.append(stat("Median years to unicorn",   f"{median_yrs:.0f} years"))
lines.append(stat("Reached status in ≤2 years",f"{fastest_n} companies ({fastest_pct:.1f}%)"))
lines.append(stat("Slowest company",
                  f"{slowest['company']} ({int(slowest['years_to_unicorn'])} years)"))
lines.append(stat("Fastest industry (median)", f"{fastest_industry} ({fastest_ind_yrs:.1f} yrs)"))
lines.append(stat("Slowest industry (median)", f"{slowest_industry} ({slowest_ind_yrs:.1f} yrs)"))
lines.append("")
lines.append(bullet(
    f"The typical path to unicorn status takes {median_yrs:.0f} years from "
    f"founding. Companies in {fastest_industry} get there fastest at a "
    f"median of {fastest_ind_yrs:.1f} years, while {slowest_industry} "
    f"companies take nearly {slowest_ind_yrs:.0f} years — likely reflecting "
    f"the longer capital cycles and regulatory environments in that sector."
))
lines.append(bullet(
    f"{fastest_pct:.1f}% of unicorns achieved the milestone within 2 years "
    f"of founding, which likely reflects companies that pivoted into "
    f"hypergrowth markets or benefited from pre-existing founder networks."
))

# ── 5. Funding Efficiency
lines.append(section("5. Funding Efficiency"))
lines.append(stat("Most capital-efficient industry",
                  f"{most_eff_industry} ({most_eff_val:.1f}x avg multiplier)"))
lines.append("")
lines.append("  Top 5 most capital-efficient companies:")
for _, row in top_eff.iterrows():
    lines.append(
        f"    {row['company']:<28} "
        f"${row['valuation_b']:.0f}B val  "
        f"${row['funding_b']:.2f}B raised  "
        f"({row['funding_efficiency']:.1f}x)"
    )
lines.append("")
lines.append(bullet(
    f"{most_eff_industry} generates the highest valuation return per dollar "
    f"raised ({most_eff_val:.1f}x on average), meaning investors in this "
    f"space have historically seen the greatest valuation leverage relative "
    f"to capital deployed."
))
lines.append(bullet(
    "Funding efficiency varies dramatically across industries, making it "
    "a stronger signal of capital allocation quality than raw valuation "
    "figures alone — and a metric worth highlighting in any investor-facing "
    "analysis."
))

# ── 6. Growth Trends
lines.append(section("6. Growth Trends"))
lines.append(stat("Data spans",               f"{first_year} – {last_year}"))
lines.append(stat("Peak year for new unicorns",f"{peak_year} ({peak_count} new companies)"))
lines.append(stat("Post-2015 vs pre-2015",    f"{decade_growth:.1f}x more unicorns"))
lines.append("")
lines.append(bullet(
    f"{peak_year} was the single biggest year for unicorn creation with "
    f"{peak_count} new entrants — likely driven by the post-pandemic surge "
    f"in venture capital activity, low interest rates, and a wave of "
    f"digital-first businesses that scaled rapidly during lockdowns."
))
lines.append(bullet(
    f"The {decade_growth:.1f}x acceleration in unicorn creation post-2015 "
    f"compared to prior years reflects both the maturing of the global "
    f"VC ecosystem and the compounding effect of earlier cohorts of "
    f"infrastructure companies enabling faster startup growth."
))

# ── Footer
lines.append("\n" + "=" * W)
lines.append("  END OF FINDINGS REPORT")
lines.append(f"  Generated from {total:,} unicorn companies across "
             f"{n_countries} countries")
lines.append("=" * W)

# ── Print & save ──────────────────────────────────────────────────────────────
report = "\n".join(lines)
print(report)

with open("report/findingsReport.txt", "w") as f:
    f.write(report)

print(f"\nSaved → report/findingsReport.txt")