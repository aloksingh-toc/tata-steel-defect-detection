# Defect Detection in Hot Rolling — Tata Steel AI Hackathon

Detect **Alpha defects** in steel coils during Hot Rolling (binary classification),
on a severely imbalanced dataset (19.5 : 1). My submission scored **64.47** on the
HackerEarth leaderboard.

> **Note on data:** The competition dataset (`dataset/`) is **not** included in this
> repository — it is the property of Tata Steel / HackerEarth and may not be
> redistributed. To run the pipeline, place `train.csv`, `test.csv`, and
> `sample_submission.csv`  into a `dataset/` folder at the project root.

## Quickstart

```bash
pip install -r requirements.txt
python run.py                      # generates outputs/expected_submission.csv
python run.py --metric f1          # alternative threshold strategy
```

## Project Structure

```
.
├── dataset/                     # raw data (train, test, sample_submission)
├── src/
│   ├── config.py                # paths, column names, CV + model hyperparameters
│   ├── features.py              # feature engineering (49 → 68 features)
│   ├── preprocess.py            # IterativeImputer → StandardScaler pipeline
│   ├── model.py                 # 4-model CV training + blend
│   ├── evaluate.py              # threshold selection + reporting
│   └── predict.py               # inference path for new data
├── scripts/
│   └── make_notebook.py         # regenerates the analysis notebook
├── notebooks/
│   └── defect_detection.ipynb   # full EDA + modelling walkthrough
├── outputs/
│   └── expected_submission.csv  # THE PREDICTION FILE
├── models/
│   └── preprocessor.pkl         # fitted imputer + scaler
├── run.py                       # end-to-end entry point
├── APPROACH.md                  # methodology + results writeup
└── requirements.txt
```

## Approach (summary)

| Step | Method |
|------|--------|
| Feature Engineering | log transforms, ratios, row-wise stats, interactions, missingness flags |
| Imputation | `IterativeImputer` (multivariate) |
| Scaling | `StandardScaler` |
| Imbalance | `scale_pos_weight` (LGB/XGB), `class_weight='balanced'` (RF/ET) |
| Models | LightGBM (Optuna-tuned) + XGBoost + Random Forest + Extra Trees |
| Ensemble | equal-weight blend |
| Threshold | min OOF probability of any defect → **Recall = 100%** |

## Results (5-Fold OOF)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8653 |
| PR-AUC | 0.3660 |
| Recall @ chosen threshold | 100% |

See [APPROACH.md](APPROACH.md) for the full methodology and an honest analysis of the
precision/recall tradeoff against the competition's `Recall=100%, Precision>90%` target.
