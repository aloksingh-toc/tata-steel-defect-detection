from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR  = ROOT / "outputs"
MODELS_DIR  = ROOT / "models"

TRAIN_PATH  = DATASET_DIR / "train.csv"
TEST_PATH   = DATASET_DIR / "test.csv"
OUTPUT_PATH = OUTPUT_DIR  / "expected_submission.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# ── Column names ───────────────────────────────────────────────────────────
ID_COL     = "CoilID"
TARGET_COL = "Y"

# ── CV ─────────────────────────────────────────────────────────────────────
N_SPLITS     = 5
RANDOM_STATE = 42

# ── Model hyperparameters (Optuna-tuned) ───────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=757,
    learning_rate=0.0898,
    max_depth=6,
    num_leaves=62,
    min_child_samples=34,
    subsample=0.605,
    colsample_bytree=0.644,
    reg_alpha=0.000207,
    reg_lambda=0.00987,
    random_state=RANDOM_STATE,
    verbose=-1,
    feature_name="auto",   # suppresses "fitted with feature names" warning
)

XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    verbosity=0,
)

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

ET_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
