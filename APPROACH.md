# Approach & Methodology — Defect Detection in Hot Rolling

## Problem

Detect the rare **Alpha defect** in steel coils produced during Hot Rolling.
Binary classification with severe class imbalance.

- **Train:** 1,352 coils, 49 process features (X1–X49), target `Y`
- **Test:** 339 coils
- **Class balance:** 1,286 non-defect vs 66 defect (**19.5 : 1**)
- **Evaluation target:** Recall = 100% (zero missed defects) and Precision > 90%

## Data Understanding

- No duplicate `CoilID`s; no train/test overlap.
- Missing values concentrated in `X15` (12%), `X42`, `X48` — missingness treated as a signal.
- `X34–X38` are large count features (X35 max ≈ 17.7M) — heavily right-skewed.
- Top correlates with `Y`: X35, X13, X36, X34, X10 (all |r| ≈ 0.24–0.26 — individually weak).

## Feature Engineering (49 → 68 features)

1. **Missingness flags** for X15, X42, X48.
2. **Log transforms** of skewed count features X34–X38.
3. **Ratio features**: X34/X35, X36/X37, X34/X36.
4. **Row-wise statistics** over the temperature group (X1–X9) and ratio group (X41–X49).
5. **Interaction terms**: X35×X13, X10×X30, X34×X13.

## Preprocessing

- **IterativeImputer** (multivariate) — better than median for X15's 12% missingness.
- **StandardScaler**.
- Fitted pipeline persisted to `models/preprocessor.pkl` for reproducible inference.

## Modelling

Four complementary models trained with **5-fold StratifiedKFold** CV, each producing
out-of-fold (OOF) and averaged test predictions:

| Model | OOF ROC-AUC |
|-------|-------------|
| LightGBM (Optuna-tuned, 50 trials) | 0.8550 |
| XGBoost | 0.8553 |
| Random Forest | 0.8551 |
| Extra Trees | 0.8575 |
| **Equal-weight blend** | **0.8653** |

Imbalance handled via `scale_pos_weight` (LGB/XGB) and `class_weight='balanced'` (RF/ET).

## Threshold Selection

Because the metric demands **zero false negatives**, the decision threshold is set to the
**minimum OOF probability assigned to any true defect** (0.004758). This guarantees
Recall = 100% on the training data. On the test set this flags 259/339 coils.

## Honest Result vs Target

| | Achieved | Required |
|---|---|---|
| Recall | **100%** ✅ | 100% |
| Precision @ Recall=100% | **6%** | >90% |
| Recall @ Precision=90% | 1.5% | 100% |

### Why 90% precision is not reachable with the current data

To allow Recall=100% with Precision>90%, the model may emit at most **7 false positives**
(66 / 0.9 − 66 ≈ 7), i.e. a false-positive rate ≤ 0.54%. At that FPR, a model with
AUC=0.865 captures only ~55% of defects. Achieving 100% recall at that FPR would require
**AUC ≈ 0.997**.

The 10 hardest defects receive OOF probabilities < 0.10 — they are statistically
indistinguishable from normal coils across all 68 features. We validated this is a **data
ceiling, not a modelling gap** by testing One-Class SVM, Isolation Forest, KNN,
Mahalanobis distance, polynomial features, neural nets, CatBoost, and stacking — none
exceeded ~8% precision at full recall (a Decision Tree reached 31% in-sample but collapsed
to 5% out-of-fold, confirming overfitting).

### What would close the gap

1. More labelled defect examples (66 is far too few for this separation).
2. Domain-specific features identifying the Alpha-defect process signature.
3. Raw sensor time-series (X1–X49 are pre-aggregated) and/or temporal ordering features.

## Tools

Python 3 · pandas · numpy · scikit-learn · LightGBM · XGBoost · Optuna · joblib
