"""
Phase 3, Step 2 — Python EDA Charts
Unicorn Companies Project
------------------------------------------------
Input:  unicorn_clean.csv
Output: charts/  folder — 6 publication-quality PNG charts

Run:  python3 notebook/eda_charts.py

Charts:
  01 — Valuation distribution (log scale histogram + box plot)
  02 — Top 15 countries by unicorn count (horizontal bar)
  03 — Industry breakdown — count vs avg valuation (dual axis)
  04 — Funding raised vs valuation (scatter, coloured by continent)
  05 — Unicorns created per year (area chart)
  06 — Years to unicorn by industry (box plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "sql/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/unicorn_companies_clean.csv", parse_dates=["date_joined"])
print(f"Loaded {len(df):,} rows\n")

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE = {
    "dark":    "#1A3C34",
    "mid":     "#2D6A4F",
    "accent":  "#52B788",
    "light":   "#D8F3DC",
    "gray":    "#6C757D",
    "bg":      "#F8F9FA",
    "text":    "#212529",
}

CONTINENT_COLORS = {
    "North America": "#2D6A4F",
    "Asia":          "#52B788",
    "Europe":        "#74C69D",
    "South America": "#B7E4C7",
    "Oceania":       "#95D5B2",
    "Africa":        "#1B4332",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DEE2E6",
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       PALETTE["gray"],
    "ytick.color":       PALETTE["gray"],
    "font.family":       "DejaVu Sans",
    "text.color":        PALETTE["text"],
    "grid.color":        "#DEE2E6",
    "grid.linewidth":    0.6,
})

def save_chart(fig, name):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved → {path}")

# ── Chart 01: Valuation distribution ─────────────────────────────────────────
print("Chart 01 — Valuation distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Unicorn valuation distribution", fontsize=15, fontweight="bold",
             color=PALETTE["dark"], y=1.02)

# Left: log-scale histogram
ax = axes[0]
ax.hist(np.log10(df["valuation_b"]), bins=30,
        color=PALETTE["accent"], edgecolor="white", linewidth=0.5, alpha=0.9)
ax.set_xlabel("Valuation (log₁₀ $B)")
ax.set_ylabel("Number of companies")
ax.set_title("Log-scale histogram")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${10**x:.0f}B"
))
ax.yaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)

# Right: box plot by continent
ax2 = axes[1]
continent_order = (df.groupby("continent")["valuation_b"]
                   .median().sort_values(ascending=False).index.tolist())
data_by_continent = [df[df["continent"] == c]["valuation_b"].values
                     for c in continent_order]
bp = ax2.boxplot(data_by_continent, patch_artist=True, vert=True,
                 medianprops=dict(color=PALETTE["dark"], linewidth=2),
                 whiskerprops=dict(color=PALETTE["gray"]),
                 capprops=dict(color=PALETTE["gray"]),
                 flierprops=dict(marker=".", color=PALETTE["gray"],
                                 markersize=3, alpha=0.5))
for patch, continent in zip(bp["boxes"], continent_order):
    patch.set_facecolor(CONTINENT_COLORS.get(continent, PALETTE["accent"]))
    patch.set_alpha(0.8)
ax2.set_yscale("log")
ax2.set_xticks(range(1, len(continent_order)+1))
ax2.set_xticklabels(continent_order, rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("Valuation ($B, log scale)")
ax2.set_title("Box plot by continent")
ax2.yaxis.grid(True, alpha=0.5)
ax2.set_axisbelow(True)

fig.tight_layout()
save_chart(fig, "01_valuation_distribution")

# ── Chart 02: Top 15 countries ────────────────────────────────────────────────
print("Chart 02 — Top 15 countries...")
top_countries = (df.groupby("country")["company"]
                 .count().sort_values(ascending=True).tail(15))

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top_countries.index, top_countries.values,
               color=PALETTE["accent"], edgecolor="white", linewidth=0.5)

# Value labels on bars
for bar in bars:
    w = bar.get_width()
    ax.text(w + 4, bar.get_y() + bar.get_height()/2,
            f"{int(w):,}", va="center", ha="left",
            fontsize=9, color=PALETTE["text"])

# Highlight top 3
for bar in bars[-3:]:
    bar.set_color(PALETTE["mid"])

ax.set_xlabel("Number of unicorn companies")
ax.set_title("Top 15 countries by unicorn count")
ax.xaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0, top_countries.max() * 1.15)
fig.tight_layout()
save_chart(fig, "02_top_countries")

# ── Chart 03: Industry — count vs avg valuation (dual axis) ──────────────────
print("Chart 03 — Industry breakdown...")
industry_stats = (df.groupby("industry")
                  .agg(count=("company", "count"),
                       avg_val=("valuation_b", "mean"))
                  .sort_values("count", ascending=True))

# Clean up duplicate AI label
industry_stats.index = industry_stats.index.str.replace(
    "Artificial Intelligence", "AI (alt label)", regex=False
)

fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twiny()

y_pos = range(len(industry_stats))
bars1 = ax1.barh(list(y_pos), industry_stats["count"],
                 color=PALETTE["light"], edgecolor=PALETTE["accent"],
                 linewidth=0.8, label="# companies")

ax2.scatter(industry_stats["avg_val"], list(y_pos),
            color=PALETTE["mid"], zorder=5, s=60, label="Avg valuation ($B)")

ax1.set_yticks(list(y_pos))
ax1.set_yticklabels(industry_stats.index, fontsize=9)
ax1.set_xlabel("Number of companies", color=PALETTE["accent"])
ax2.set_xlabel("Avg valuation ($B)", color=PALETTE["mid"])
ax1.tick_params(axis="x", colors=PALETTE["accent"])
ax2.tick_params(axis="x", colors=PALETTE["mid"])
ax1.set_title("Industry breakdown — company count vs avg valuation")
ax1.xaxis.grid(True, alpha=0.4)
ax1.set_axisbelow(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

fig.tight_layout()
save_chart(fig, "03_industry_breakdown")

# ── Chart 04: Funding vs valuation scatter ────────────────────────────────────
print("Chart 04 — Funding vs valuation scatter...")
plot_df = df[(df["funding_b"] > 0) & (df["valuation_b"] > 0)].copy()

fig, ax = plt.subplots(figsize=(11, 7))

for continent, grp in plot_df.groupby("continent"):
    ax.scatter(grp["funding_b"], grp["valuation_b"],
               label=continent, alpha=0.65, s=28,
               color=CONTINENT_COLORS.get(continent, PALETTE["gray"]))

# Annotate top 8 by valuation
top8 = plot_df.nlargest(8, "valuation_b")
for _, row in top8.iterrows():
    ax.annotate(row["company"],
                xy=(row["funding_b"], row["valuation_b"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7.5, color=PALETTE["dark"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["gray"],
                                lw=0.6))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Funding raised ($B, log scale)")
ax.set_ylabel("Valuation ($B, log scale)")
ax.set_title("Funding raised vs valuation — coloured by continent")
ax.xaxis.grid(True, alpha=0.4)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Continent", fontsize=8, title_fontsize=9,
          framealpha=0.8, loc="upper left")

fig.tight_layout()
save_chart(fig, "04_funding_vs_valuation_scatter")

# ── Chart 05: Unicorns per year (area chart) ──────────────────────────────────
print("Chart 05 — Unicorns per year...")
yearly = (df.groupby("year_joined")
          .agg(count=("company", "count"),
               avg_val=("valuation_b", "mean"))
          .dropna()
          .reset_index()
          .sort_values("year_joined"))
yearly["year_joined"] = yearly["year_joined"].astype(int)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.fill_between(yearly["year_joined"], yearly["count"],
                 alpha=0.25, color=PALETTE["accent"])
ax1.plot(yearly["year_joined"], yearly["count"],
         color=PALETTE["mid"], linewidth=2.5, marker="o",
         markersize=5, label="New unicorns")

ax2.plot(yearly["year_joined"], yearly["avg_val"],
         color=PALETTE["dark"], linewidth=1.8, linestyle="--",
         marker="s", markersize=4, label="Avg valuation ($B)")

ax1.set_xlabel("Year")
ax1.set_ylabel("New unicorns", color=PALETTE["mid"])
ax2.set_ylabel("Avg valuation ($B)", color=PALETTE["dark"])
ax1.tick_params(axis="y", colors=PALETTE["mid"])
ax2.tick_params(axis="y", colors=PALETTE["dark"])
ax1.set_title("Unicorn creation over time")
ax1.xaxis.grid(True, alpha=0.4)
ax1.set_axisbelow(True)
ax1.set_xticks(yearly["year_joined"])
ax1.set_xticklabels(yearly["year_joined"].astype(str), rotation=45, ha="right")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=9)

fig.tight_layout()
save_chart(fig, "05_unicorns_per_year")

# ── Chart 06: Years to unicorn by industry (box plot) ────────────────────────
print("Chart 06 — Years to unicorn by industry...")
industry_order = (df.groupby("industry")["years_to_unicorn"]
                  .median().sort_values().index.tolist())

plot_data = [df[df["industry"] == ind]["years_to_unicorn"].dropna().values
             for ind in industry_order]

# Clean industry labels for display
clean_labels = [
    ind.replace("E-commerce & direct-to-consumer", "E-commerce")
       .replace("Supply chain, logistics, & delivery", "Supply chain")
       .replace("Internet software & services", "Internet software")
       .replace("Mobile & telecommunications", "Mobile & telecom")
       .replace("Data management & analytics", "Data mgmt & analytics")
    for ind in industry_order
]

fig, ax = plt.subplots(figsize=(12, 7))
bp = ax.boxplot(plot_data, patch_artist=True, vert=False,
                medianprops=dict(color=PALETTE["dark"], linewidth=2.2),
                whiskerprops=dict(color=PALETTE["gray"], linewidth=1),
                capprops=dict(color=PALETTE["gray"], linewidth=1),
                flierprops=dict(marker=".", color=PALETTE["gray"],
                                markersize=3, alpha=0.4))

# Gradient fill from lightest to darkest
n = len(bp["boxes"])
greens = plt.cm.Greens(np.linspace(0.25, 0.75, n))
for patch, color in zip(bp["boxes"], greens):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_yticks(range(1, len(clean_labels) + 1))
ax.set_yticklabels(clean_labels, fontsize=9)
ax.set_xlabel("Years from founding to unicorn status")
ax.set_title("Time to unicorn status by industry")
ax.xaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
save_chart(fig, "06_years_to_unicorn_by_industry")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\nAll 6 charts saved to: {OUTPUT_DIR}/")