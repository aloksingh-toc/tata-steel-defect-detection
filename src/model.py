import logging
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from src.config import N_SPLITS, RANDOM_STATE, LGBM_PARAMS, XGB_PARAMS, RF_PARAMS, ET_PARAMS

log = logging.getLogger(__name__)


def _build_models(scale_pos_weight: float) -> dict[str, Any]:
    return {
        "lgbm": lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight),
        "xgb":  xgb.XGBClassifier(**XGB_PARAMS,   scale_pos_weight=scale_pos_weight),
        "rf":   RandomForestClassifier(**RF_PARAMS),
        "et":   ExtraTreesClassifier(**ET_PARAMS),
    }


def train_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    blend_weights: dict[str, float] | None = None,
) -> dict:
    """
    Train each model with stratified K-fold CV, collect OOF and test predictions,
    then blend into final arrays.

    Returns a dict with keys:
        oof_blend, test_blend, oof_preds, test_preds
    """
    counts    = np.bincount(y.astype(int))
    scale_pos = float(counts[0]) / float(counts[1])
    models    = _build_models(scale_pos)
    cv        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds:  dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}

    for name, model in models.items():
        oof    = np.zeros(len(y))
        test_p = np.zeros(len(X_test))

        for tr, va in cv.split(X, y):
            m = clone(model)          # sklearn-safe clone — no get_params hack
            m.fit(X[tr], y[tr])
            oof[va]  = m.predict_proba(X[va])[:, 1]
            test_p  += m.predict_proba(X_test)[:, 1] / N_SPLITS

        auc = roc_auc_score(y, oof)
        ap  = average_precision_score(y, oof)
        log.info("  %-6s | AUC: %.4f | PR-AUC: %.4f", name, auc, ap)

        oof_preds[name]  = oof
        test_preds[name] = test_p

    # Weighted or equal blend
    names = list(models.keys())
    if blend_weights is None:
        weights = np.ones(len(names)) / len(names)
    else:
        weights = np.array([blend_weights.get(n, 1.0) for n in names])
        weights /= weights.sum()

    oof_blend  = np.average(list(oof_preds.values()),  axis=0, weights=weights)
    test_blend = np.average(list(test_preds.values()), axis=0, weights=weights)

    auc = roc_auc_score(y, oof_blend)
    ap  = average_precision_score(y, oof_blend)
    log.info("  %-6s | AUC: %.4f | PR-AUC: %.4f  ← FINAL", "blend", auc, ap)

    return {
        "oof_blend":   oof_blend,
        "test_blend":  test_blend,
        "oof_preds":   oof_preds,
        "test_preds":  test_preds,
    }
