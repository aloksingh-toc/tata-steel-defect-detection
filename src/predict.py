"""
Inference module — apply trained preprocessor to new unlabelled data.
Usage:
    from src.predict import predict_new
    preds = predict_new(new_df, preprocessor_path, blend_result, threshold)
"""
import numpy as np
import pandas as pd

from src.config    import PREPROCESSOR_PATH, ID_COL
from src.features  import engineer, get_feature_cols
from src.preprocess import load_preprocessor


def predict_new(
    df: pd.DataFrame,
    blend_test: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Given test predictions (already computed by train_and_predict),
    apply threshold and return a submission-ready DataFrame.
    """
    final_preds = (blend_test >= threshold).astype(int)
    return pd.DataFrame({ID_COL: df[ID_COL].values.astype(int), "Y": final_preds})


def predict_from_saved(
    df: pd.DataFrame,
    model_results: dict,
    threshold: float,
) -> pd.DataFrame:
    """
    Full inference path using saved preprocessor + already-computed blend.
    Loads the fitted preprocessor from disk, engineers + transforms features,
    then applies the threshold to produce predictions.
    """
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    df_fe        = engineer(df)
    feature_cols = get_feature_cols(df_fe)
    X            = preprocessor.transform(df_fe[feature_cols].values)

    # Re-use blend from the training run if available, otherwise use saved test_blend
    test_blend = model_results.get("test_blend")
    if test_blend is None:
        raise ValueError("model_results must contain 'test_blend' key.")

    return predict_new(df, test_blend, threshold)
