import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


def find_threshold(
    oof: np.ndarray,
    y: np.ndarray,
    metric: str = "recall_100",
) -> float:
    """
    Find optimal threshold from OOF predictions.

    metric options:
        "recall_100" — lowest threshold that achieves Recall = 100% (zero FN)
        "f1"         — threshold that maximises F1 score
        "precision_recall_balance" — threshold where precision ≈ recall
    """
    if metric == "recall_100":
        pos_probs = oof[y == 1]
        if len(pos_probs) == 0:
            raise ValueError("No positive samples in y.")
        return float(pos_probs.min())

    best_thresh = 0.5
    best_score  = -np.inf   # works for all metrics including negative-valued ones
    for t in np.arange(0.01, 0.99, 0.005):
        preds = (oof >= t).astype(int)
        if preds.sum() == 0:
            continue
        if metric == "f1":
            score = f1_score(y, preds)
        elif metric == "precision_recall_balance":
            r = recall_score(y, preds)
            p = precision_score(y, preds)
            score = -abs(p - r)   # closer to 0 → better; needs -inf start
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        if score > best_score:
            best_score, best_thresh = score, t

    return float(best_thresh)


def threshold_report(oof: np.ndarray, y: np.ndarray, selected: float | None = None) -> None:
    """Log a recall/precision table across a range of thresholds."""
    import logging
    log = logging.getLogger(__name__)
    header = f"\n{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>6} {'TP':>5} {'FP':>5} {'Predicted':>10}"
    log.info(header)
    log.info("-" * 62)
    for t in np.arange(0.001, 0.51, 0.02):
        preds = (oof >= t).astype(int)
        if preds.sum() == 0:
            continue
        rec  = recall_score(y, preds)
        prec = precision_score(y, preds)
        f1   = f1_score(y, preds)
        tp   = int((preds * y).sum())
        fp   = int(preds.sum() - tp)
        marker = " ←" if selected is not None and abs(t - selected) < 0.011 else ""
        log.info("%10.3f %8.3f %10.3f %6.3f %5d %5d %10d%s",
                 t, rec, prec, f1, tp, fp, preds.sum(), marker)
