"""
Phase 4, Step 6 — Model Comparison Report
Unicorn Companies Project
------------------------------------------------
Input:  ML_outputs/step1_regression_results.csv
        ML_outputs/step2_classifier_results.csv
        ML_outputs/step3_multiclass_results.csv
        ML_outputs/step4_cluster_assignments.csv
        ML_outputs/step5_forecast_results.csv

Output: ML_outputs/model_results.csv
        ML_outputs/step6_model_comparison.png
        Prints full report to terminal

Run:  python3 notebook/model_comparison.py

Aggregates all model results into a single summary table
and generates a visual comparison chart.

Run this AFTER steps 1–5 have completed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("ML_outputs", exist_ok=True)

DARK   = "#1A3C34"
MID    = "#2D6A4F"
ACCENT = "#52B788"
LIGHT  = "#D8F3DC"
GRAY   = "#6C757D"
BG     = "#F8F9FA"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DEE2E6",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "font.family":       "DejaVu Sans",
    "grid.color":        "#DEE2E6",
    "grid.linewidth":    0.6,
})

W = 68   # terminal width

def section(title):
    return f"\n{'═' * W}\n  {title}\n{'═' * W}"

def load(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded: {path}  ({len(df)} rows)")
        return df
    else:
        print(f"  MISSING: {path} — run the corresponding step first")
        return None

print("=" * W)
print("  PHASE 4 — MODEL COMPARISON REPORT")
print("  Unicorn Companies Project")
print("=" * W)
print("\nLoading results from ML_outputs/...")

# ── 1. Load all result files ──────────────────────────────────────────────────
reg_df  = load("ML_outputs/step1_regression_results.csv",    "Regression")
cls_df  = load("ML_outputs/step2_classifier_results.csv",    "Binary Classifier")
mc_df   = load("ML_outputs/step3_multiclass_results.csv",    "Multiclass Classifier")
clust   = load("ML_outputs/step4_cluster_assignments.csv",   "Clustering")
fcast   = load("ML_outputs/step5_forecast_results.csv",      "Forecast")

# ── 2. Regression summary ─────────────────────────────────────────────────────
print(section("1. Valuation Prediction — Regression"))
if reg_df is not None:
    print(f"\n  {'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV R²':>10}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for _, row in reg_df.iterrows():
        best_marker = " ◀ best" if row["r2"] == reg_df["r2"].max() else ""
        print(f"  {row['model']:<22} {row['rmse']:>8.4f} {row['mae']:>8.4f} "
              f"{row['r2']:>8.4f} {row['cv_r2_mean']:>10.4f}{best_marker}")

    best_reg = reg_df.loc[reg_df["r2"].idxmax()]
    print(f"\n  Best model  : {best_reg['model']}")
    print(f"  R²          : {best_reg['r2']} (explains {best_reg['r2']*100:.0f}% of variance in log-valuation)")
    print(f"  RMSE        : {best_reg['rmse']} (in log units — back-transform with np.expm1() for $B)")
    print(f"\n  Key insight : Funding raised and industry are the strongest predictors.")
    print(f"  Portfolio note: Show the feature importance chart — it's the most")
    print(f"  visually compelling output from this model for a non-technical audience.")

# ── 3. Binary classifier summary ─────────────────────────────────────────────
print(section("2. High-Value Unicorn Classifier (≥$10B)"))
if cls_df is not None:
    print(f"\n  {'Model':<22} {'F1':>8} {'ROC-AUC':>9} {'Avg Prec':>10} {'CV F1':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*10} {'─'*8}")
    for _, row in cls_df.iterrows():
        best_marker = " ◀ best" if row["roc_auc"] == cls_df["roc_auc"].max() else ""
        print(f"  {row['model']:<22} {row['f1']:>8.4f} {row['roc_auc']:>9.4f} "
              f"{row['avg_precision']:>10.4f} {row['cv_f1_mean']:>8.4f}{best_marker}")

    best_cls = cls_df.loc[cls_df["roc_auc"].idxmax()]
    print(f"\n  Best model  : {best_cls['model']}")
    print(f"  ROC-AUC     : {best_cls['roc_auc']} (1.0 = perfect, 0.5 = random)")
    print(f"\n  Key insight : Class imbalance (94%/6%) is the main challenge.")
    print(f"  class_weight='balanced' and scale_pos_weight address this.")
    print(f"  Portfolio note: Use the ROC curve chart — it shows model discrimination")
    print(f"  power clearly even when the class split is extreme.")

# ── 4. Multiclass classifier summary ─────────────────────────────────────────
print(section("3. Continent Classifier (6 Classes)"))
if mc_df is not None:
    print(f"\n  {'Model':<22} {'Accuracy':>10} {'F1-macro':>10} {'F1-weighted':>12}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*12}")
    for _, row in mc_df.iterrows():
        best_marker = " ◀ best" if row["f1_macro"] == mc_df["f1_macro"].max() else ""
        print(f"  {row['model']:<22} {row['accuracy']:>10.4f} "
              f"{row['f1_macro']:>10.4f} {row['f1_weighted']:>12.4f}{best_marker}")

    best_mc = mc_df.loc[mc_df["f1_macro"].idxmax()]
    print(f"\n  Best model  : {best_mc['model']}")
    print(f"  F1-macro    : {best_mc['f1_macro']} (penalises Africa/Oceania misclassification)")
    print(f"  F1-weighted : {best_mc['f1_weighted']} (more practical for imbalanced classes)")
    print(f"\n  Key insight : North America and Asia are reliably predicted.")
    print(f"  Africa (n=3) and Oceania (n=8) are too small for reliable classification.")
    print(f"  Portfolio note: Be upfront about this in your README — showing awareness")
    print(f"  of data limitations is a mark of a good analyst.")

# ── 5. Clustering summary ─────────────────────────────────────────────────────
print(section("4. K-Means Clustering"))
if clust is not None:
    cluster_profile = (clust.groupby(["cluster", "archetype"])
                       .agg(count=("company", "count"),
                            avg_val=("valuation_b", "mean"),
                            avg_fund=("funding_b", "mean"),
                            avg_yrs=("years_to_unicorn", "mean"))
                       .reset_index()
                       .round(2))
    print(f"\n  {'Cluster':<4} {'Archetype':<22} {'Count':>6} "
          f"{'Avg Val':>9} {'Avg Fund':>10} {'Avg Yrs':>9}")
    print(f"  {'─'*4} {'─'*22} {'─'*6} {'─'*9} {'─'*10} {'─'*9}")
    for _, row in cluster_profile.iterrows():
        print(f"  {int(row['cluster']):<4} {row['archetype']:<22} "
              f"{int(row['count']):>6} {row['avg_val']:>9.1f} "
              f"{row['avg_fund']:>10.2f} {row['avg_yrs']:>9.1f}")

    print(f"\n  Key insight : K-Means reveals distinct unicorn archetypes.")
    print(f"  Portfolio note: The PCA scatter + archetype labels is the most")
    print(f"  storytelling-friendly output in the whole project. Lead with this")
    print(f"  in Tableau and your README.")

# ── 6. Forecast summary ───────────────────────────────────────────────────────
print(section("5. Unicorn Creation Forecast (Prophet)"))
if fcast is not None:
    fcast["month"] = pd.to_datetime(fcast["month"])
    future_fcast = fcast[fcast["month"].dt.year >= 2023]
    yearly = (future_fcast.groupby(future_fcast["month"].dt.year)
              .agg(predicted=("predicted_unicorns", "sum"),
                   lower=("lower_90", "sum"),
                   upper=("upper_90", "sum"))
              .round(0).astype(int))
    print(f"\n  {'Year':>6} {'Predicted':>12} {'Lower 90%':>12} {'Upper 90%':>12}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12}")
    for year, row in yearly.iterrows():
        print(f"  {year:>6} {row['predicted']:>12} "
              f"{row['lower']:>12} {row['upper']:>12}")

    print(f"\n  Key insight : The 2021 spike (520 unicorns) is a known anomaly.")
    print(f"  Prophet's forecast reverts to the long-run trend post-surge.")
    print(f"  Portfolio note: Always mention the 2021 anomaly when presenting")
    print(f"  the forecast — it shows domain awareness of the VC cycle.")

# ── 7. Comparison chart ───────────────────────────────────────────────────────
print(section("Generating comparison chart..."))

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Phase 4 — Model results summary",
             fontsize=14, fontweight="bold", color=DARK, y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

model_color_map = {
    "Linear Regression":   GRAY,
    "Logistic Regression": GRAY,
    "Random Forest":       ACCENT,
    "XGBoost":             MID,
}

# Panel 1: Regression R²
if reg_df is not None:
    ax = fig.add_subplot(gs[0, 0])
    colors = [model_color_map.get(m, ACCENT) for m in reg_df["model"]]
    bars = ax.bar(reg_df["model"], reg_df["r2"], color=colors,
                  edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, reg_df["r2"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, min(1.05, reg_df["r2"].max() * 1.2))
    ax.set_title("Regression — R²\n(valuation prediction)")
    ax.set_xticklabels(reg_df["model"], rotation=15, ha="right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

# Panel 2: Binary ROC-AUC
if cls_df is not None:
    ax = fig.add_subplot(gs[0, 1])
    colors = [model_color_map.get(m, ACCENT) for m in cls_df["model"]]
    bars = ax.bar(cls_df["model"], cls_df["roc_auc"], color=colors,
                  edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1,
               alpha=0.5, label="Random (0.5)")
    for bar, val in zip(bars, cls_df["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Binary classifier — ROC-AUC\n(high-value unicorn)")
    ax.set_xticklabels(cls_df["model"], rotation=15, ha="right", fontsize=8)
    ax.legend(fontsize=7.5)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

# Panel 3: Multiclass F1
if mc_df is not None:
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(mc_df))
    w = 0.35
    colors_macro    = [model_color_map.get(m, ACCENT) for m in mc_df["model"]]
    colors_weighted = [MID] * len(mc_df)
    bars1 = ax.bar(x - w/2, mc_df["f1_macro"],    w, label="F1-macro",
                   color=colors_macro, edgecolor="white", linewidth=0.4)
    bars2 = ax.bar(x + w/2, mc_df["f1_weighted"], w, label="F1-weighted",
                   color=[LIGHT]*len(mc_df), edgecolor=MID, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mc_df["model"], rotation=15, ha="right", fontsize=8)
    ax.set_title("Multiclass classifier — F1\n(continent prediction)")
    ax.legend(fontsize=7.5)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

# Panel 4: Regression RMSE comparison
if reg_df is not None:
    ax = fig.add_subplot(gs[1, 0])
    colors = [model_color_map.get(m, ACCENT) for m in reg_df["model"]]
    bars = ax.bar(reg_df["model"], reg_df["rmse"], color=colors,
                  edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, reg_df["rmse"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Regression — RMSE\n(lower is better)")
    ax.set_xticklabels(reg_df["model"], rotation=15, ha="right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

# Panel 5: Binary classifier F1 vs Avg Precision
if cls_df is not None:
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(cls_df))
    w = 0.35
    ax.bar(x - w/2, cls_df["f1"],            w, label="F1 score",
           color=[model_color_map.get(m, ACCENT) for m in cls_df["model"]],
           edgecolor="white", linewidth=0.4)
    ax.bar(x + w/2, cls_df["avg_precision"], w, label="Avg precision",
           color=[LIGHT]*len(cls_df), edgecolor=MID, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_df["model"], rotation=15, ha="right", fontsize=8)
    ax.set_title("Binary classifier\nF1 vs average precision")
    ax.legend(fontsize=7.5)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)

# Panel 6: Summary text
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
summary_lines = [
    "Model summary",
    "",
    "Regression:",
    f"  Best R² achieved with XGBoost",
    f"  Target: log(valuation $B)",
    "",
    "Binary classifier:",
    f"  Best ROC-AUC with XGBoost",
    f"  Handles 94/6 class split",
    "",
    "Multiclass:",
    f"  6 continent classes",
    f"  Low-n classes flagged",
    "",
    "Clustering:",
    f"  K archetypes identified",
    f"  PCA visualisation",
    "",
    "Forecast:",
    f"  Prophet — 3-yr horizon",
    f"  2021 surge noted",
]
ax.text(0.05, 0.97, "\n".join(summary_lines),
        transform=ax.transAxes, fontsize=8.5, va="top",
        color=DARK, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                  edgecolor=ACCENT, linewidth=1))

fig.savefig("ml_outputs/step6_model_comparison.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved → ml_outputs/step6_model_comparison.png")

# ── 8. Master CSV ──────────────────────────────────────────────────────────────
master_rows = []
if reg_df is not None:
    for _, row in reg_df.iterrows():
        master_rows.append({
            "phase": "Step 1 — Regression",
            "model": row["model"],
            "primary_metric": "R²",
            "primary_value": row["r2"],
            "secondary_metric": "RMSE",
            "secondary_value": row["rmse"],
            "cv_score": row["cv_r2_mean"],
        })
if cls_df is not None:
    for _, row in cls_df.iterrows():
        master_rows.append({
            "phase": "Step 2 — Binary Classifier",
            "model": row["model"],
            "primary_metric": "ROC-AUC",
            "primary_value": row["roc_auc"],
            "secondary_metric": "F1",
            "secondary_value": row["f1"],
            "cv_score": row["cv_f1_mean"],
        })
if mc_df is not None:
    for _, row in mc_df.iterrows():
        master_rows.append({
            "phase": "Step 3 — Multiclass Classifier",
            "model": row["model"],
            "primary_metric": "F1-macro",
            "primary_value": row["f1_macro"],
            "secondary_metric": "Accuracy",
            "secondary_value": row["accuracy"],
            "cv_score": None,
        })

master_df = pd.DataFrame(master_rows)
master_df.to_csv("ml_outputs/model_results.csv", index=False)
print("  Saved → ml_outputs/model_results.csv")

# ── 9. Final summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * W}")
print("  PHASE 4 COMPLETE")
print(f"{'=' * W}")
print("  Files in ml_outputs/:")
for f in sorted(os.listdir("ml_outputs")):
    size = os.path.getsize(f"ml_outputs/{f}")
    print(f"    {f:<45} {size:>8,} bytes")
print(f"\n  Total ML outputs: {len(os.listdir('ml_outputs'))} files")
print(f"{'=' * W}")