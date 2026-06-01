import joblib
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def build_preprocessor() -> Pipeline:
    """Return an unfitted impute→scale pipeline."""
    return Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=RANDOM_STATE, initial_strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def save_preprocessor(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_preprocessor(path: Path) -> Pipeline:
    return joblib.load(path)
