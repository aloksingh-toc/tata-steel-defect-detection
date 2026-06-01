import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET_COL

# ── Column group constants ─────────────────────────────────────────────────
TEMP_COLS         = [f"X{i}" for i in range(1, 10)]
RATIO_COLS        = [f"X{i}" for i in range(41, 50)]
SKEWED_COLS       = ["X34", "X35", "X36", "X37", "X38"]
MISSING_FLAG_COLS = ["X15", "X42", "X48"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    required = set(TEMP_COLS + RATIO_COLS + SKEWED_COLS + MISSING_FLAG_COLS + ["X13", "X10", "X30"])
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"engineer(): input DataFrame is missing columns: {sorted(missing_cols)}")

    df = df.copy()

    # Missingness indicators — X15 has 12% missing; flag is informative
    for col in MISSING_FLAG_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    # Log-transform skewed large-integer features (X35 max = 17.7M)
    for col in SKEWED_COLS:
        df[f"{col}_log"] = np.log1p(df[col])

    # Ratio features between count columns
    df["X34_X35_ratio"] = np.log1p(df["X34"]) / (np.log1p(df["X35"]) + 1e-9)
    df["X36_X37_ratio"] = df["X36"] / (df["X37"] + 1e-9)
    df["X34_X36_ratio"] = df["X34"] / (df["X36"] + 1e-9)

    # Row-wise statistics on temperature process group (uses module constant)
    df["temp_mean"]  = df[TEMP_COLS].mean(axis=1)
    df["temp_std"]   = df[TEMP_COLS].std(axis=1)
    df["temp_range"] = df[TEMP_COLS].max(axis=1) - df[TEMP_COLS].min(axis=1)

    # Row-wise statistics on ratio/fraction group (uses module constant)
    df["ratio_mean"] = df[RATIO_COLS].mean(axis=1)
    df["ratio_sum"]  = df[RATIO_COLS].sum(axis=1)

    # Interaction features between top-correlated columns
    df["X35_X13"] = np.log1p(df["X35"]) * df["X13"]
    df["X10_X30"] = df["X10"] * df["X30"]
    df["X34_X13"] = np.log1p(df["X34"]) * df["X13"]

    return df


def get_feature_cols(df: pd.DataFrame, exclude: tuple[str, ...] = (ID_COL, TARGET_COL)) -> list[str]:
    return [c for c in df.columns if c not in exclude]
