"""
Phase 4, Step 1 — Valuation Prediction (Regression)
Unicorn Companies Project
------------------------------------------------
Input: data/unicorn_companies_clean.csv
Output: ML_outputs/regression_results.csv
        ML_outputs/feature_importance.png
        ML_outputs/actual_vs_predicted.png

Run:  python3 notebook/regression.py

Models trained:
  - Linear Regression     (baseline)
  - Random Forest         (ensemble)
  - XGBoost               (gradient boosting)

Target:  log_valuation  (log of valuation_b — reduces skew)
Features: funding_b, year_founded, years_to_unicorn,
          funding_efficiency, industry (encoded),
          continent (encoded), month_joined, quarter_joined
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import OneHotEncoder, StandardScaler
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LinearRegression
from sklearn.ensemble           import RandomForestRegressor
from sklearn.metrics            import (mean_squared_error,
                                        mean_absolute_error, r2_score)
from xgboost                    import XGBRegressor

os.makedirs("ML_outputs", exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
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

FEATURES_NUM = ["funding_b", "year_founded", "years_to_unicorn",
                 "funding_efficiency", "month_joined", "quarter_joined"]
FEATURES_CAT = ["industry", "continent"]
TARGET       = "log_valuation"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set : {len(X_train):,} rows")
print(f"Test set     : {len(X_test):,} rows")
print(f"Target       : {TARGET}  (log of valuation $B)\n")

# ── 2. Preprocessing pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                              FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False),            FEATURES_CAT),
])

# ── 3. Define models ──────────────────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ("prep",  preprocessor),
        ("model", LinearRegression()),
    ]),
    "Random Forest": Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=8,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )),
    ]),
    "XGBoost": Pipeline([
        ("prep",  preprocessor),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )),
    ]),
}

# ── 4. Train, evaluate, cross-validate ───────────────────────────────────────
print("=" * 60)
print(f"{'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'CV R² (5-fold)':>16}")
print("=" * 60)

results     = []
predictions = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    print(f"{name:<22} {rmse:>7.4f} {mae:>7.4f} {r2:>7.4f} "
          f"  {cv_mean:.4f} ± {cv_std:.4f}")

    results.append({
        "model":        name,
        "rmse":         round(rmse, 4),
        "mae":          round(mae,  4),
        "r2":           round(r2,   4),
        "cv_r2_mean":   round(cv_mean, 4),
        "cv_r2_std":    round(cv_std,  4),
    })
    predictions[name] = y_pred

print("=" * 60)

# ── 5. Feature importance (Random Forest) ────────────────────────────────────
rf_pipe    = models["Random Forest"]
rf_model   = rf_pipe.named_steps["model"]
ohe_cats   = (rf_pipe.named_steps["prep"]
              .named_transformers_["cat"]
              .get_feature_names_out(FEATURES_CAT))
feat_names = FEATURES_NUM + list(ohe_cats)
importances = rf_model.feature_importances_

# Aggregate OHE features back to their parent category
imp_series = pd.Series(importances, index=feat_names)
agg_imp    = {}
for f in FEATURES_NUM:
    agg_imp[f] = imp_series[f]
for cat in FEATURES_CAT:
    agg_imp[cat] = imp_series[
        [i for i in feat_names if i.startswith(cat + "_")]
    ].sum()

imp_df = (pd.Series(agg_imp)
          .sort_values(ascending=True)
          .reset_index())
imp_df.columns = ["feature", "importance"]

# Clean feature names for display
display_map = {
    "funding_b":          "Funding raised ($B)",
    "year_founded":       "Year founded",
    "years_to_unicorn":   "Years to unicorn",
    "funding_efficiency": "Funding efficiency",
    "month_joined":       "Month joined",
    "quarter_joined":     "Quarter joined",
    "industry":           "Industry",
    "continent":          "Continent",
}
imp_df["feature"] = imp_df["feature"].map(display_map)

print(f"\nTop features (Random Forest):")
for _, row in imp_df.sort_values("importance", ascending=False).iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"  {row['feature']:<26} {row['importance']:.4f}  {bar}")

# ── 6. Charts ─────────────────────────────────────────────────────────────────

# Chart A: Feature importance
fig, ax = plt.subplots(figsize=(9, 5))
colors = [ACCENT if i == len(imp_df) - 1 else MID
          for i in range(len(imp_df))]
bars = ax.barh(imp_df["feature"], imp_df["importance"],
               color=colors, edgecolor="white", linewidth=0.4)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", ha="left", fontsize=8.5,
            color=DARK)
ax.set_xlabel("Feature importance (aggregated)", color=GRAY)
ax.set_title("Random Forest — feature importance for valuation prediction")
ax.xaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("ml_outputs/step1_feature_importance.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nSaved → ml_outputs/step1_feature_importance.png")

# Chart B: Actual vs predicted (all 3 models)
fig = plt.figure(figsize=(14, 4.5))
fig.suptitle("Actual vs predicted log-valuation — test set",
             fontsize=13, fontweight="bold", color=DARK, y=1.02)
gs = gridspec.GridSpec(1, 3, figure=fig)

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = fig.add_subplot(gs[idx])
    ax.scatter(y_test, y_pred, alpha=0.45, s=18,
               color=ACCENT, edgecolors=MID, linewidths=0.3)
    lo = min(y_test.min(), y_pred.min()) - 0.1
    hi = max(y_test.max(), y_pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=DARK,
            linewidth=1.4, linestyle="--", label="Perfect fit")
    r2_val = results[idx]["r2"]
    ax.set_title(f"{name}\nR² = {r2_val:.3f}")
    ax.set_xlabel("Actual log-valuation")
    ax.set_ylabel("Predicted log-valuation")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.grid(True, alpha=0.4)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("ml_outputs/step1_actual_vs_predicted.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ML_outputs/step1_actual_vs_predicted.png")

# ── 7. Save results ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("ML_outputs/step1_regression_results.csv", index=False)
print("Saved → ML_outputs/step1_regression_results.csv")

print("\n── Best model ──────────────────────────────────────────────")
best = results_df.loc[results_df["r2"].idxmax()]
print(f"  {best['model']}  |  R² = {best['r2']}  |  RMSE = {best['rmse']}")
print("Note: target was log_valuation — back-transform with np.expm1() for $B values")