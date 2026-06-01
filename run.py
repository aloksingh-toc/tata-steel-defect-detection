"""
Entry point — runs the full pipeline end-to-end.
Usage: python run.py [--metric recall_100|f1]
"""
import argparse
import logging

import pandas as pd

from src.config    import TRAIN_PATH, TEST_PATH, OUTPUT_PATH, PREPROCESSOR_PATH, ID_COL, TARGET_COL
from src.features  import engineer, get_feature_cols
from src.preprocess import build_preprocessor, save_preprocessor
from src.model     import train_and_predict
from src.evaluate  import find_threshold, threshold_report

log = logging.getLogger(__name__)


def main(metric: str = "recall_100") -> None:
    # ── Directories ────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    log.info("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    log.info("  Train: %s  |  Test: %s", train.shape, test.shape)

    # ── Feature Engineering ────────────────────────────────────────────────
    log.info("Engineering features...")
    train_fe     = engineer(train)
    test_fe      = engineer(test)
    feature_cols = get_feature_cols(train_fe)
    log.info("  %d features (was 49)", len(feature_cols))

    y        = train[TARGET_COL].values
    test_ids = test[ID_COL].values

    # ── Preprocessing ──────────────────────────────────────────────────────
    log.info("Preprocessing (impute → scale)...")
    prep   = build_preprocessor()
    X      = prep.fit_transform(train_fe[feature_cols].values)
    X_test = prep.transform(test_fe[feature_cols].values)
    save_preprocessor(prep, PREPROCESSOR_PATH)
    log.info("  Preprocessor saved → %s", PREPROCESSOR_PATH)

    # ── Train ──────────────────────────────────────────────────────────────
    log.info("Training models (5-fold CV)...")
    results = train_and_predict(X, y, X_test)

    # ── Threshold ──────────────────────────────────────────────────────────
    threshold = find_threshold(results["oof_blend"], y, metric=metric)
    log.info("Threshold analysis (selected=%.6f, metric=%s):", threshold, metric)
    threshold_report(results["oof_blend"], y, selected=threshold)

    # ── Predict & Save ─────────────────────────────────────────────────────
    final_preds = (results["test_blend"] >= threshold).astype(int)
    log.info("Predicted defective: %d / %d", final_preds.sum(), len(final_preds))

    submission = pd.DataFrame({ID_COL: test_ids.astype(int), TARGET_COL: final_preds})
    submission.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved: %s", OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Defect Detection Pipeline")
    parser.add_argument(
        "--metric",
        default="recall_100",
        choices=["recall_100", "f1", "precision_recall_balance"],
        help="Threshold optimisation metric (default: recall_100)",
    )
    args = parser.parse_args()
    main(metric=args.metric)
