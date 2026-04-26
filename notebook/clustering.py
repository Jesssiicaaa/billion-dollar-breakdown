"""
Phase 4, Step 4 — Unicorn Clustering (Unsupervised)
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: ML_outputs/step4_elbow_silhouette.png
        ML_outputs/step4_cluster_pca.png
        ML_outputs/step4_cluster_profiles.png
        ML_outputs/step4_cluster_assignments.csv

Run:  python3 notebook/clustering.py

Method: K-Means clustering
  - Elbow method + silhouette scores to choose optimal K
  - PCA (2 components) for 2-D visualisation
  - Cluster profiling to name and describe each archetype
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.cluster            import KMeans
from sklearn.decomposition      import PCA
from sklearn.metrics            import silhouette_score
from sklearn.pipeline           import Pipeline

os.makedirs("ml_outputs", exist_ok=True)

DARK   = "#1A3C34"
MID    = "#2D6A4F"
ACCENT = "#52B788"
LIGHT  = "#D8F3DC"
GRAY   = "#6C757D"
BG     = "#F8F9FA"

CLUSTER_PALETTE = ["#2D6A4F", "#52B788", "#95D5B2", "#D8F3DC",
                   "#1B4332", "#B7E4C7"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DEE2E6",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "font.family":       "DejaVu Sans",
    "grid.color":        "#DEE2E6",
    "grid.linewidth":    0.6,
})

# ── 1. Load & prepare ─────────────────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies_clean.csv")
df["industry"] = df["industry"].replace(
    "Artificial Intelligence", "Artificial intelligence"
)

FEATURES_NUM = ["valuation_b", "funding_b", "years_to_unicorn",
                 "funding_efficiency", "year_founded"]
FEATURES_CAT = ["industry", "continent"]

X = df[FEATURES_NUM + FEATURES_CAT].copy()

# Cap funding_efficiency at 99th percentile to prevent outlier-dominated clusters
# (e.g. Zapier with $0 funding recorded skews the entire feature space)
cap_val = X["funding_efficiency"].quantile(0.99)
X["funding_efficiency"] = X["funding_efficiency"].clip(upper=cap_val)
print(f"Capped funding_efficiency at 99th percentile: {cap_val:.1f}x")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                              FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False),            FEATURES_CAT),
])
X_processed = preprocessor.fit_transform(X)
print(f"Processed feature matrix: {X_processed.shape}")

# ── 3. Elbow method + silhouette scores ───────────────────────────────────────
print("\nRunning elbow + silhouette analysis (K = 2 to 10)...")
K_range    = range(2, 11)
inertias   = []
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=15,
                random_state=42, max_iter=500)
    labels = km.fit_predict(X_processed)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_processed, labels,
                                       sample_size=None))
    print(f"  K={k:2d}  inertia={km.inertia_:10.1f}  "
          f"silhouette={sil_scores[-1]:.4f}")

best_k = K_range[np.argmax(sil_scores)]
print(f"\nOptimal K by silhouette: {best_k}")

# ── 4. Fit final model ────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20,
                  random_state=42, max_iter=500)
df["cluster"] = km_final.fit_predict(X_processed)

cluster_counts = df["cluster"].value_counts().sort_index()
print(f"\nCluster sizes:")
for c, n in cluster_counts.items():
    print(f"  Cluster {c}: {n} companies")

# ── 5. PCA for visualisation ──────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)
df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]
var_explained = pca.explained_variance_ratio_
print(f"\nPCA variance explained: {var_explained[0]*100:.1f}% + "
      f"{var_explained[1]*100:.1f}% = "
      f"{sum(var_explained)*100:.1f}% total")

# ── 6. Cluster profiles ───────────────────────────────────────────────────────
profile_cols = ["valuation_b", "funding_b", "years_to_unicorn",
                "funding_efficiency", "year_founded", "is_high_value"]
profiles = df.groupby("cluster")[profile_cols].mean().round(2)
profiles["count"] = cluster_counts

print("\nCluster profiles (mean values):")
print(profiles.to_string())

# Assign descriptive archetype names based on profiles
def name_cluster(row):
    if row["valuation_b"] >= 8:
        return "Mega-unicorns"
    elif row["funding_efficiency"] >= 15:
        return "Capital-efficient"
    elif row["years_to_unicorn"] <= 4:
        return "Fast risers"
    elif row["funding_b"] >= 1.0:
        return "VC-heavy"
    else:
        return "Steady growers"

profiles["archetype"] = profiles.apply(name_cluster, axis=1)
# Fallback: if any two archetypes share the same name, append cluster id
seen = {}
for idx in profiles.index:
    name = profiles.loc[idx, "archetype"]
    if name in seen:
        profiles.loc[idx,       "archetype"] = f"{name} ({idx})"
        profiles.loc[seen[name], "archetype"] = f"{name} ({seen[name]})"
    seen[name] = idx

df["archetype"] = df["cluster"].map(profiles["archetype"])
print(f"\nArchetype names: {profiles['archetype'].tolist()}")

# ── 7. Charts ─────────────────────────────────────────────────────────────────

# Chart A: Elbow + silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Choosing optimal K — elbow method & silhouette scores",
             fontsize=13, fontweight="bold", color=DARK, y=1.02)

ax1.plot(list(K_range), inertias, "o-", color=MID, linewidth=2,
         markersize=6)
ax1.axvline(best_k, color=ACCENT, linestyle="--", linewidth=1.5,
            label=f"Chosen K={best_k}")
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("Inertia (within-cluster sum of squares)")
ax1.set_title("Elbow method")
ax1.legend(fontsize=9)
ax1.xaxis.grid(True, alpha=0.4)
ax1.yaxis.grid(True, alpha=0.4)
ax1.set_axisbelow(True)

ax2.plot(list(K_range), sil_scores, "s-", color=MID, linewidth=2,
         markersize=6)
ax2.axvline(best_k, color=ACCENT, linestyle="--", linewidth=1.5,
            label=f"Best K={best_k} ({max(sil_scores):.3f})")
ax2.set_xlabel("Number of clusters (K)")
ax2.set_ylabel("Silhouette score")
ax2.set_title("Silhouette scores")
ax2.legend(fontsize=9)
ax2.xaxis.grid(True, alpha=0.4)
ax2.yaxis.grid(True, alpha=0.4)
ax2.set_axisbelow(True)

fig.tight_layout()
fig.savefig("ml_outputs/step4_elbow_silhouette.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nSaved → ml_outputs/step4_elbow_silhouette.png")

# Chart B: PCA scatter coloured by cluster
fig, ax = plt.subplots(figsize=(10, 7))
for c in sorted(df["cluster"].unique()):
    mask  = df["cluster"] == c
    label = profiles.loc[c, "archetype"]
    n     = cluster_counts[c]
    ax.scatter(df.loc[mask, "pca_1"], df.loc[mask, "pca_2"],
               label=f"Cluster {c}: {label} (n={n})",
               color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
               alpha=0.65, s=28, edgecolors="white", linewidths=0.3)

# Annotate top 5 by valuation
top5 = df.nlargest(5, "valuation_b")
for _, row in top5.iterrows():
    ax.annotate(row["company"],
                xy=(row["pca_1"], row["pca_2"]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=7.5, color=DARK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5))

ax.set_xlabel(f"PCA component 1  ({var_explained[0]*100:.1f}% variance)")
ax.set_ylabel(f"PCA component 2  ({var_explained[1]*100:.1f}% variance)")
ax.set_title(f"K-Means clusters (K={best_k}) — PCA 2-D projection")
ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
ax.xaxis.grid(True, alpha=0.4)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("ml_outputs/step4_cluster_pca.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ml_outputs/step4_cluster_pca.png")

# Chart C: Cluster profile radar / bar comparison
metrics   = ["valuation_b", "funding_b", "years_to_unicorn", "funding_efficiency"]
labels    = ["Avg valuation ($B)", "Avg funding ($B)",
             "Avg years to unicorn", "Avg funding efficiency"]

fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
fig.suptitle("Cluster profiles — key metric comparison",
             fontsize=13, fontweight="bold", color=DARK, y=1.02)

for ax, metric, label in zip(axes, metrics, labels):
    vals   = [profiles.loc[c, metric] for c in sorted(df["cluster"].unique())]
    names  = [profiles.loc[c, "archetype"] for c in sorted(df["cluster"].unique())]
    colors = [CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
              for c in sorted(df["cluster"].unique())]
    bars   = ax.bar(range(len(vals)), vals, color=colors,
                    edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names],
                       fontsize=7.5, ha="center")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7.5, color=DARK)

fig.tight_layout()
fig.savefig("ml_outputs/step4_cluster_profiles.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ml_outputs/step4_cluster_profiles.png")

# ── 8. Save assignments ───────────────────────────────────────────────────────
output_cols = ["company", "industry", "country", "continent",
               "valuation_b", "funding_b", "years_to_unicorn",
               "cluster", "archetype"]
df[output_cols].to_csv("ml_outputs/step4_cluster_assignments.csv", index=False)
print("Saved → ml_outputs/step4_cluster_assignments.csv")

print("\n── Cluster summary ─────────────────────────────────────────")
print(profiles[["archetype", "count", "valuation_b", "funding_b",
                "years_to_unicorn", "funding_efficiency"]].to_string())