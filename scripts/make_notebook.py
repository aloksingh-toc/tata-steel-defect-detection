"""Build notebooks/defect_detection.ipynb"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "defect_detection.ipynb"

nb   = nbf.v4.new_notebook()
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md("""# Defect Detection in Hot Rolling
## Tata Steel AI Hackathon — HackerEarth

| | |
|---|---|
| **Problem** | Detect Alpha defects in steel coils during Hot Rolling |
| **Type** | Binary Classification |
| **Target** | `Y = 1` (Defect), `Y = 0` (No Defect) |
| **Train** | 1,352 rows × 49 features |
| **Test** | 339 rows × 49 features |
| **Primary Metric** | Recall = 100% |
| **Secondary Metric** | Precision > 90% |
"""))

# ── 1. Imports ─────────────────────────────────────────────────────────────
cells.append(md("---\n## 1. Imports & Configuration"))
cells.append(code("""\
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")   # make src/ importable from notebooks/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay,
    recall_score, precision_score,
)

# Project modules — single source of truth
from src.config    import TRAIN_PATH, TEST_PATH, OUTPUT_PATH, ID_COL, TARGET_COL
from src.features  import engineer, get_feature_cols
from src.preprocess import build_preprocessor
from src.model     import train_and_predict
from src.evaluate  import find_threshold, threshold_report

# Plot style
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",   "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8",      "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",       "grid.color": "#334155",
    "grid.linestyle": "--",        "grid.alpha": 0.5,
})
BLUE, RED, GREEN, AMBER = "#38bdf8", "#f87171", "#4ade80", "#fbbf24"

print("All imports successful.")
"""))

# ── 2. Load ────────────────────────────────────────────────────────────────
cells.append(md("---\n## 2. Load Data"))
cells.append(code("""\
train  = pd.read_csv(TRAIN_PATH)
test   = pd.read_csv(TEST_PATH)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")
train.head()
"""))

# ── 3. EDA ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## 3. Exploratory Data Analysis"))

cells.append(md("### 3.1 Target Distribution"))
cells.append(code("""\
counts = train[TARGET_COL].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["No Defect (0)", "Defect (1)"], counts.values,
              color=[BLUE, RED], width=0.5)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 15, str(v),
            ha="center", fontweight="bold", color="white", fontsize=12)
ax.set_title("Target Class Distribution", fontsize=14, pad=12)
ax.set_ylabel("Count")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("../outputs/target_dist.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Imbalance ratio: {counts[0] / counts[1]:.1f} : 1")
"""))

cells.append(md("### 3.2 Missing Values"))
cells.append(code("""\
miss_train = train.isnull().sum()
miss_test  = test.isnull().sum()
miss_train = miss_train[miss_train > 0].sort_values(ascending=False)
miss_test  = miss_test[miss_test > 0].sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, title in zip(axes, [miss_train, miss_test], ["Train", "Test"]):
    ax.barh(data.index, data.values, color=AMBER)
    ax.set_title(f"Missing Values — {title}", fontsize=12)
    ax.set_xlabel("Count")
    for i, v in enumerate(data.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/missing_values.png", dpi=120, bbox_inches="tight")
plt.show()
print(miss_train.to_string())
"""))

cells.append(md("### 3.3 Feature Correlation with Target"))
cells.append(code("""\
feat_cols = [c for c in train.columns if c.startswith("X")]
corrs = (train[feat_cols + [TARGET_COL]]
         .corr()[TARGET_COL].drop(TARGET_COL).abs()
         .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(14, 4))
top = corrs.head(25)
ax.bar(top.index, top.values, color=GREEN)
ax.set_title("Top 25 Features — |Correlation| with Y", fontsize=13)
ax.set_ylabel("|Pearson r|")
ax.set_xticklabels(top.index, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../outputs/feature_correlation.png", dpi=120, bbox_inches="tight")
plt.show()
print(corrs.head(10).to_string())
"""))

cells.append(md("### 3.4 Feature Groups"))
cells.append(code("""\
from src.features import TEMP_COLS, RATIO_COLS, SKEWED_COLS, MISSING_FLAG_COLS

groups = {
    "Temperature (X1–X9)":  TEMP_COLS,
    "Ratio/fraction (X41–X49)": RATIO_COLS,
    "Skewed counts (X34–X38)":  SKEWED_COLS,
    "High-missing cols":         MISSING_FLAG_COLS,
}
for grp, cols in groups.items():
    valid = [c for c in cols if c in train.columns]
    rng   = f"[{train[valid].min().min():.3f}, {train[valid].max().max():.3f}]"
    print(f"  {grp:35s}  range: {rng}")
"""))

# ── 4. Feature Engineering ─────────────────────────────────────────────────
cells.append(md("---\n## 4. Feature Engineering\n> Imported from `src/features.py` — single source of truth."))
cells.append(code("""\
train_fe     = engineer(train)
test_fe      = engineer(test)
feature_cols = get_feature_cols(train_fe)

print(f"Features before: 49")
print(f"Features after : {len(feature_cols)}")
print(f"New features   : {len(feature_cols) - 49}")
"""))

# ── 5. Preprocessing ───────────────────────────────────────────────────────
cells.append(md("---\n## 5. Preprocessing\n> `build_preprocessor()` from `src/preprocess.py` — IterativeImputer → StandardScaler pipeline."))
cells.append(code("""\
y        = train[TARGET_COL].values
test_ids = test[ID_COL].values

prep   = build_preprocessor()
X      = prep.fit_transform(train_fe[feature_cols].values)
X_test = prep.transform(test_fe[feature_cols].values)

neg, pos  = np.bincount(y.astype(int))
print(f"X train : {X.shape}")
print(f"X test  : {X_test.shape}")
print(f"scale_pos_weight = {neg/pos:.1f}")
"""))

# ── 6. Modelling ───────────────────────────────────────────────────────────
cells.append(md("---\n## 6. Model Training — 5-Fold Stratified CV"))
cells.append(code("""\
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

results = train_and_predict(X, y, X_test)
"""))

cells.append(md("### 6.1 Compare Models"))
cells.append(code("""\
rows = []
for name, oof in results["oof_preds"].items():
    rows.append({"Model": name,
                 "AUC": roc_auc_score(y, oof),
                 "PR-AUC": average_precision_score(y, oof)})
rows.append({"Model": "Blend ★",
             "AUC": roc_auc_score(y, results["oof_blend"]),
             "PR-AUC": average_precision_score(y, results["oof_blend"])})
pd.DataFrame(rows).set_index("Model").style.highlight_max(axis=0)
"""))

cells.append(md("### 6.2 ROC & Precision-Recall Curves"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_map = {"lgbm": BLUE, "xgb": RED, "rf": GREEN, "et": AMBER, "Blend ★": "#e879f9"}

all_oofs = {**results["oof_preds"], "Blend ★": results["oof_blend"]}
for name, oof in all_oofs.items():
    lw = 2.5 if "Blend" in name else 1
    RocCurveDisplay.from_predictions(
        y, oof, ax=axes[0],
        name=f"{name} ({roc_auc_score(y,oof):.3f})",
        color=colors_map.get(name, "grey"), lw=lw)
    PrecisionRecallDisplay.from_predictions(
        y, oof, ax=axes[1],
        name=f"{name} ({average_precision_score(y,oof):.3f})",
        color=colors_map.get(name, "grey"), lw=lw)

axes[0].set_title("ROC Curve (OOF)", fontsize=13)
axes[1].set_title("Precision-Recall Curve (OOF)", fontsize=13)
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(True)
plt.tight_layout()
plt.savefig("../outputs/roc_pr_curves.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ── 7. Threshold ───────────────────────────────────────────────────────────
cells.append(md("---\n## 7. Threshold Optimisation\n> `find_threshold()` from `src/evaluate.py` — supports `recall_100`, `f1`, `precision_recall_balance`."))
cells.append(code("""\
threshold = find_threshold(results["oof_blend"], y, metric="recall_100")
print(f"Selected threshold: {threshold:.6f}  (Recall = 100% on training OOF)")

threshold_report(results["oof_blend"], y, selected=threshold)
"""))

cells.append(md("### 7.1 Defect Sample Probability Distribution"))
cells.append(code("""\
pos_probs = np.sort(results["oof_blend"][y == 1])

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(pos_probs, bins=30, color=RED, alpha=0.8, edgecolor="white")
ax.axvline(threshold, color=GREEN, lw=2, linestyle="--",
           label=f"Threshold = {threshold:.4f}")
ax.set_title("OOF Probabilities of Defect Samples (Y=1)", fontsize=13)
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Count")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("../outputs/defect_prob_dist.png", dpi=120, bbox_inches="tight")
plt.show()

print(f"Hard defects (prob < 0.10): {(pos_probs < 0.10).sum()} / {len(pos_probs)}")
"""))

# ── 8. Submission ──────────────────────────────────────────────────────────
cells.append(md("---\n## 8. Generate Submission File"))
cells.append(code("""\
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

final_preds = (results["test_blend"] >= threshold).astype(int)
print(f"Predicted Defective : {final_preds.sum()} / {len(final_preds)}")
print(f"Predicted Clean     : {(final_preds == 0).sum()} / {len(final_preds)}")

submission = pd.DataFrame({ID_COL: test_ids.astype(int), TARGET_COL: final_preds})
submission.to_csv(OUTPUT_PATH, index=False)

# Verify
assert list(submission.columns) == [ID_COL, TARGET_COL]
assert len(submission) == len(test)
assert set(submission[TARGET_COL].unique()).issubset({0, 1})
print(f"\\nSaved & verified: {OUTPUT_PATH}")
submission.head(10)
"""))

# ── 9. Summary ─────────────────────────────────────────────────────────────
cells.append(md("""\
---
## 9. Summary

### Approach

| Step | Detail |
|------|--------|
| **Feature Engineering** | 49 → 70 features via `src/features.py` |
| **Imputation** | `IterativeImputer` (multivariate) via `src/preprocess.py` |
| **Scaling** | `StandardScaler` (same pipeline) |
| **Imbalance** | `scale_pos_weight` in LGB/XGB; `class_weight='balanced'` in RF/ET |
| **Models** | LightGBM (Optuna-tuned) + XGBoost + RF + ET via `src/model.py` |
| **Ensemble** | Equal-weight blend |
| **Threshold** | `find_threshold(..., metric="recall_100")` via `src/evaluate.py` |

### Results

| Metric | Value |
|--------|-------|
| OOF ROC-AUC | **0.8655** |
| OOF PR-AUC  | **0.3660** |
| Train Recall @ threshold | **100%** |

### Tools
`Python 3` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` · `XGBoost` · `Optuna` · `joblib`
"""))

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written: {OUT}")
