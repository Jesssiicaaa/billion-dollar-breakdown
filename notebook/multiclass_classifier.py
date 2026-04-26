"""
Phase 4, Step 3 — Continent Classifier (Multiclass)
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: ML_outputs/multiclass_results.csv
        ML_outputs/confusion_matrix_heatmap.png
        ML_outputs/per_class_f1.png

Run:  python3 notebook/multiclass_classifier.py

Target:  continent  (6 classes — North America, Asia, Europe,
                     South America, Oceania, Africa)

Note on class imbalance:
  Africa (3) and Oceania (8) have very few samples.
  The model will note this honestly — low sample classes
  will have high uncertainty and are clearly flagged.

Models trained:
  - Logistic Regression   (baseline)
  - Random Forest         (ensemble, class_weight=balanced)
  - XGBoost               (gradient boosting)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        f1_score, accuracy_score)
from xgboost                    import XGBClassifier

os.makedirs("ML_outputs", exist_ok=True)

DARK   = "#1A3C34"
MID    = "#2D6A4F"
ACCENT = "#52B788"
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
                 "funding_efficiency", "valuation_b", "month_joined"]
FEATURES_CAT = ["industry"]
TARGET       = "continent"

print("Class distribution:")
counts = df[TARGET].value_counts()
for cont, n in counts.items():
    bar = "█" * (n // 10)
    flag = "  ⚠ very few samples" if n < 15 else ""
    print(f"  {cont:<18} {n:>4}  {bar}{flag}")
print()
print("Note: Africa (3) and Oceania (8) have too few samples for")
print("reliable classification. Results for these classes should")
print("be interpreted with caution.\n")

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

# Encode labels to integers for XGBoost
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"Training set : {len(X_train):,} rows")
print(f"Test set     : {len(X_test):,} rows\n")

# ── 2. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                              FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False),            FEATURES_CAT),
])

# ── 3. Models ─────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("prep",  preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000, C=1.0, random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced",
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )),
    ]),
    "XGBoost": Pipeline([
        ("prep",  preprocessor),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            num_class=len(class_names),
            objective="multi:softprob",
            random_state=42, verbosity=0, eval_metric="mlogloss"
        )),
    ]),
}

# ── 4. Train & evaluate ───────────────────────────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
pred_store = {}

print("=" * 65)
print(f"{'Model':<22} {'Accuracy':>9} {'F1-macro':>9} {'F1-weighted':>12}")
print("=" * 65)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro",   zero_division=0)
    f1_weighted= f1_score(y_test, y_pred, average="weighted",zero_division=0)

    print(f"{name:<22} {acc:>9.4f} {f1_macro:>9.4f} {f1_weighted:>12.4f}")

    results.append({
        "model":       name,
        "accuracy":    round(acc,         4),
        "f1_macro":    round(f1_macro,    4),
        "f1_weighted": round(f1_weighted, 4),
    })
    pred_store[name] = y_pred

print("=" * 65)

# Per-class report for best model
best_name = max(results, key=lambda r: r["f1_macro"])["model"]
best_pred = pred_store[best_name]
print(f"\nPer-class report — {best_name} (best F1-macro):")
present_labels = sorted(np.unique(np.concatenate([y_test, best_pred])))
present_names  = [class_names[i] for i in present_labels]
print(classification_report(y_test, best_pred,
                             labels=present_labels,
                             target_names=present_names,
                             zero_division=0))

# ── 5. Per-class F1 breakdown ─────────────────────────────────────────────────
per_class_f1 = {}
for name, y_pred in pred_store.items():
    f1s = f1_score(y_test, y_pred, average=None,
                   labels=range(len(class_names)), zero_division=0)
    per_class_f1[name] = f1s

per_class_df = pd.DataFrame(per_class_f1, index=class_names)

# ── 6. Charts ─────────────────────────────────────────────────────────────────

# Chart A: Confusion matrix heatmap (best model)
fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test, best_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="#DEE2E6",
            cbar_kws={"label": "Row-normalised proportion"},
            ax=ax)
ax.set_xlabel("Predicted continent", labelpad=10)
ax.set_ylabel("Actual continent",    labelpad=10)
ax.set_title(f"Confusion matrix — {best_name}\n"
             f"(counts shown; colour = row-normalised proportion)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=9)
fig.tight_layout()
fig.savefig("ml_outputs/step3_confusion_matrix_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nSaved → ml_outputs/step3_confusion_matrix_heatmap.png")

# Chart B: Per-class F1 grouped bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x       = np.arange(len(class_names))
width   = 0.25
colors  = [GRAY, ACCENT, MID]

for i, (model_name, color) in enumerate(zip(per_class_df.columns, colors)):
    bars = ax.bar(x + i * width, per_class_df[model_name],
                  width=width, label=model_name,
                  color=color, edgecolor="white", linewidth=0.4, alpha=0.88)

ax.set_xticks(x + width)
ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1.08)
ax.set_title("Per-class F1 score by model — continent classification")
ax.legend(fontsize=9, framealpha=0.9)
ax.yaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)

# Annotate low-sample classes
for i, cls in enumerate(class_names):
    n = counts.get(cls, 0)
    if n < 15:
        ax.annotate(f"n={n}\n(low)", xy=(x[i] + width, 0.02),
                    fontsize=7.5, ha="center", color="red", style="italic")

fig.tight_layout()
fig.savefig("ml_outputs/step3_per_class_f1.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ml_outputs/step3_per_class_f1.png")

# ── 7. Save results ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("ml_outputs/step3_multiclass_results.csv", index=False)
print("Saved → ml_outputs/step3_multiclass_results.csv")

print("\n── Best model ──────────────────────────────────────────────")
best = max(results, key=lambda r: r["f1_macro"])
print(f"  {best['model']}  |  F1-macro = {best['f1_macro']}  "
      f"|  Accuracy = {best['accuracy']}")
print("\nNote: F1-macro weights all classes equally — it penalises")
print("heavily for Africa and Oceania misclassification despite low n.")
print("F1-weighted is more forgiving and better reflects real-world utility.")