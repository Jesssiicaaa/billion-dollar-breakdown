"""
Phase 1, Step 3 — Generate 6 exploratory charts from the raw dataset
-----------------------------------------------------------------------
Reads unicorn_companies.csv and saves 6 charts to /report/phase1/

Run: python3 notebook/charts.py | tee report/chartsReport.txt
Dependencies: pip install pandas matplotlib seaborn numpy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "data/unicorn_companies.csv"
OUTPUT_DIR = "report/phase1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette
PRIMARY   = "#1A3C34"
SECONDARY = "#2D6A4F"
ACCENT    = "#52B788"
LIGHT     = "#D8F3DC"
HIGHLIGHT = "#E63946"
GRAY      = "#6C757D"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F8F9FA",
    "axes.edgecolor"    : "#DEE2E6",
    "axes.grid"         : True,
    "grid.color"        : "#DEE2E6",
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 120,
})

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["Date Joined"] = pd.to_datetime(df["Date Joined"], dayfirst=True, errors="coerce")
df["Year Joined"] = df["Date Joined"].dt.year
print(f"Loaded {len(df):,} rows. Generating charts...\n")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Valuation distribution (histogram + KDE)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 1 — Valuation Distribution", fontsize=15, fontweight="bold", y=1.01)

# Left: raw histogram (right-skewed)
ax1 = axes[0]
ax1.hist(df["Valuation ($Billions)"], bins=40, color=ACCENT, edgecolor="white", linewidth=0.5)
ax1.set_title("Raw valuation (right-skewed)")
ax1.set_xlabel("Valuation ($B)")
ax1.set_ylabel("Count")
ax1.axvline(df["Valuation ($Billions)"].median(), color=HIGHLIGHT,
            linestyle="--", linewidth=1.5, label=f"Median: ${df['Valuation ($Billions)'].median():.0f}B")
"""axis labels are too crowded, so we'll just show the median as a reference line"""
ax1.axvline(df["Valuation ($Billions)"].mean(), color=PRIMARY,
            linestyle="--", linewidth=1.5, label=f"Mean: ${df['Valuation ($Billions)'].mean():.1f}B")
"""axis labels are too crowded, so we'll just show the mean as a reference line"""
ax1.legend()

# Right: log-transformed histogram
ax2 = axes[1]
log_vals = np.log1p(df["Valuation ($Billions)"])
ax2.hist(log_vals, bins=30, color=SECONDARY, edgecolor="white", linewidth=0.5)
ax2.set_title("Log-transformed valuation (more normal)")
ax2.set_xlabel("log(1 + Valuation $B)")
ax2.set_ylabel("Count")
ax2.text(0.97, 0.95, "Note: log transform needed\nfor ML regression models",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=8, color=GRAY,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
path = f"{OUTPUT_DIR}/chart1_valuation_distribution.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Top 15 countries by unicorn count
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("Chart 2 — Top 15 Countries by Unicorn Count",
             fontsize=15, fontweight="bold")

ctry = df["Country"].value_counts().head(15).sort_values()
colors = [HIGHLIGHT if c == "United States" else
          SECONDARY  if c == "China"         else ACCENT
          for c in ctry.index]

bars = ax.barh(ctry.index, ctry.values, color=colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, ctry.values):
    ax.text(val + 3, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=9, color=PRIMARY)

ax.set_xlabel("Number of Unicorn Companies")
ax.set_xlim(0, ctry.max() * 1.12)
ax.invert_yaxis()

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=HIGHLIGHT, label="United States"),
    Patch(facecolor=SECONDARY, label="China"),
    Patch(facecolor=ACCENT,    label="Other countries"),
]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
path = f"{OUTPUT_DIR}/chart2_top_countries.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Industry breakdown (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("Chart 3 — Unicorn Companies by Industry",
             fontsize=15, fontweight="bold")

ind = df["Industry"].value_counts().sort_values()
norm = plt.Normalize(ind.min(), ind.max())
colors = plt.cm.YlGn(norm(ind.values))  # green gradient by count

bars = ax.barh(ind.index, ind.values, color=colors, edgecolor="white", linewidth=0.4)
for bar, val in zip(bars, ind.values):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9, color=PRIMARY)

ax.set_xlabel("Number of Unicorn Companies")
ax.set_xlim(0, ind.max() * 1.1)

sm = plt.cm.ScalarMappable(cmap="YlGn", norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Count", shrink=0.6)

plt.tight_layout()
path = f"{OUTPUT_DIR}/chart3_industry_breakdown.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Funding vs. Valuation scatter (log-log)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("Chart 4 — Funding vs. Valuation (log scale)",
             fontsize=15, fontweight="bold")

scatter_df = df.dropna(subset=["Funding ($ Billions)"])
continent_colors = {
    "North America" : HIGHLIGHT,
    "Asia"          : SECONDARY,
    "Europe"        : "#4895EF",
    "South America" : "#F4A261",
    "Oceania"       : "#9B5DE5",
    "Africa"        : "#FEE440",
}
for cont, grp in scatter_df.groupby("Continent"):
    ax.scatter(
        np.log1p(grp["Funding ($ Billions)"]),
        np.log1p(grp["Valuation ($Billions)"]),
        label=cont, alpha=0.65, s=28,
        color=continent_colors.get(cont, GRAY), edgecolors="none",
    )

# Annotate top 8 outliers
top8 = scatter_df.nlargest(8, "Valuation ($Billions)")
for _, row in top8.iterrows():
    ax.annotate(
        row["Company"],
        xy=(np.log1p(row["Funding ($ Billions)"]),
            np.log1p(row["Valuation ($Billions)"])),
        xytext=(5, 4), textcoords="offset points",
        fontsize=7.5, color=PRIMARY,
    )

ax.set_xlabel("log(1 + Funding $B)")
ax.set_ylabel("log(1 + Valuation $B)")
ax.legend(title="Continent", loc="upper left", framealpha=0.8)

plt.tight_layout()
path = f"{OUTPUT_DIR}/chart4_funding_vs_valuation.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — New unicorns per year (line chart)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Chart 5 — New Unicorns Created Per Year",
             fontsize=15, fontweight="bold")

yr = (
    df.dropna(subset=["Year Joined"])
    .groupby("Year Joined")
    .size()
    .reset_index(name="count")
)
yr["Year Joined"] = yr["Year Joined"].astype(int)

ax.fill_between(yr["Year Joined"], yr["count"], alpha=0.18, color=ACCENT)
ax.plot(yr["Year Joined"], yr["count"], color=SECONDARY, linewidth=2.5,
        marker="o", markersize=5, markerfacecolor=HIGHLIGHT)

peak_yr = yr.loc[yr["count"].idxmax()]
ax.annotate(
    f"Peak: {int(peak_yr['Year Joined'])}\n({int(peak_yr['count'])} companies)",
    xy=(peak_yr["Year Joined"], peak_yr["count"]),
    xytext=(peak_yr["Year Joined"] - 3, peak_yr["count"] - 30),
    fontsize=9, color=PRIMARY,
    arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.2),
)

ax.set_xlabel("Year")
ax.set_ylabel("New Unicorn Companies")
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
plt.xticks(rotation=45)

plt.tight_layout()
path = f"{OUTPUT_DIR}/chart5_unicorns_per_year.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 — Continent share (donut chart)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Chart 6 — Unicorn Share by Continent",
             fontsize=15, fontweight="bold")

cont = df["Continent"].value_counts()
donut_colors = [HIGHLIGHT, SECONDARY, "#4895EF", "#F4A261", "#9B5DE5", "#FEE440"]
wedges, texts, autotexts = ax.pie(
    cont.values,
    labels=cont.index,
    colors=donut_colors[:len(cont)],
    autopct=lambda p: f"{p:.1f}%\n({int(p/100*len(df)):,})",
    startangle=140,
    pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_color("white")

ax.text(0, 0, f"{len(df):,}\nunicorns", ha="center", va="center",
        fontsize=13, fontweight="bold", color=PRIMARY)

plt.tight_layout()
path = f"{OUTPUT_DIR}/chart6_continent_share.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")

# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\nAll 6 charts saved to: {OUTPUT_DIR}/")
print("Next step → phase1_step4_findings.py")