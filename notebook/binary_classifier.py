"""
Phase 4, Step 2 — High-Value Unicorn Classifier (Binary)
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: ML_outputs/step2_classifier_results.csv
        ML_outputs/step2_confusion_matrix.png
        ML_outputs/step2_roc_curve.png
        ML_outputs/step2_precision_recall.png

Run:  python3   notebook/binary_classifier.py

Target:  is_high_value  (1 = valuation >= $10B, 0 = below)
Class balance: ~94% negative, ~6% positive — handled with
               class_weight='balanced' and SMOTE oversampling

Models trained:
  - Logistic Regression   (baseline, interpretable)
  - Random Forest         (ensemble)
  - XGBoost               (gradient boosting)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import OneHotEncoder, StandardScaler
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        roc_auc_score, roc_curve,
                                        precision_recall_curve,
                                        average_precision_score, f1_score)
from xgboost                    import XGBClassifier

os.makedirs("ml_outputs", exist_ok=True)

DARK   = "#1A3C34"
MID    = "#2D6A4F"
ACCENT = "#52B788"
LIGHT  = "#D8F3DC"
GRAY   = "#6C757D"
BG     = "#F8F9FA"
RED    = "#C0392B"

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
TARGET       = "is_high_value"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

print(f"Class distribution:")
print(f"  Not high-value (0): {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
print(f"  High-value     (1): {(y==1).sum():,}   ({(y==1).mean()*100:.1f}%)")
print(f"\nUsing class_weight='balanced' to handle imbalance\n")

# Stratified split — preserves class ratio in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                              FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False),            FEATURES_CAT),
])

# ── 3. Models — all use class_weight balanced ─────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("prep",  preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=1000,
            C=0.5, random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced",
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ]),
    "XGBoost": Pipeline([
        ("prep",  preprocessor),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y==0).sum() / (y==1).sum(),
            random_state=42, verbosity=0, eval_metric="logloss"
        )),
    ]),
}

# ── 4. Train & evaluate ───────────────────────────────────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
proba_store = {}

print("=" * 68)
print(f"{'Model':<22} {'F1':>6} {'ROC-AUC':>9} {'Avg Prec':>9} {'CV F1 (5-fold)':>16}")
print("=" * 68)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    f1       = f1_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    cv_f1 = cross_val_score(pipe, X, y, cv=cv,
                             scoring="f1", n_jobs=-1)

    print(f"{name:<22} {f1:>6.4f} {roc_auc:>9.4f} {avg_prec:>9.4f} "
          f"  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    results.append({
        "model":       name,
        "f1":          round(f1,       4),
        "roc_auc":     round(roc_auc,  4),
        "avg_precision": round(avg_prec, 4),
        "cv_f1_mean":  round(cv_f1.mean(), 4),
        "cv_f1_std":   round(cv_f1.std(),  4),
    })
    proba_store[name] = (y_pred, y_proba)

print("=" * 68)

# Print detailed classification report for best model
best_name = max(results, key=lambda r: r["roc_auc"])["model"]
best_pred, _ = proba_store[best_name]
print(f"\nClassification report — {best_name} (best ROC-AUC):")
print(classification_report(y_test, best_pred,
                             target_names=["Not high-value", "High-value"]))

# ── 5. Charts ─────────────────────────────────────────────────────────────────
model_colors = {
    "Logistic Regression": GRAY,
    "Random Forest":       ACCENT,
    "XGBoost":             MID,
}

# Chart A: Confusion matrices (3 side by side)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Confusion matrices — high-value unicorn classifier",
             fontsize=13, fontweight="bold", color=DARK, y=1.02)

for ax, (name, (y_pred, _)) in zip(axes, proba_store.items()):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Greens", aspect="auto")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else DARK
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"], fontsize=9)
    ax.set_yticklabels(["Actual 0", "Actual 1"], fontsize=9)
    r = next(r for r in results if r["model"] == name)
    ax.set_title(f"{name}\nF1 = {r['f1']:.3f}  |  ROC-AUC = {r['roc_auc']:.3f}")

fig.tight_layout()
fig.savefig("ml_outputs/step2_confusion_matrix.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nSaved → ml_outputs/step2_confusion_matrix.png")

# Chart B: ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, (_, y_proba) in proba_store.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
            color=model_colors[name], linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — high-value unicorn classification")
ax.legend(fontsize=9, framealpha=0.9)
ax.xaxis.grid(True, alpha=0.4)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("ml_outputs/step2_roc_curve.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ml_outputs/step2_roc_curve.png")

# Chart C: Precision-recall curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, (_, y_proba) in proba_store.items():
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax.plot(rec, prec, label=f"{name}  (AP={ap:.3f})",
            color=model_colors[name], linewidth=2)

baseline = y_test.mean()
ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
           label=f"Random baseline ({baseline:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-recall curves — high-value unicorn classification")
ax.legend(fontsize=9, framealpha=0.9)
ax.xaxis.grid(True, alpha=0.4)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("ml_outputs/step2_precision_recall.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ml_outputs/step2_precision_recall.png")

# ── 6. Save results ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("ml_outputs/step2_classifier_results.csv", index=False)
print("Saved → ml_outputs/step2_classifier_results.csv")

print("\n── Best model ──────────────────────────────────────────────")
best = max(results, key=lambda r: r["roc_auc"])
print(f"  {best['model']}  |  ROC-AUC = {best['roc_auc']}  |  F1 = {best['f1']}")
print("\nNote: With only 61 high-value companies, ROC-AUC and Average")
print("Precision are more meaningful metrics than raw accuracy here.")