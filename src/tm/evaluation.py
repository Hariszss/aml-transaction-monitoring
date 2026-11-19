"""Evaluation helpers: alert counts, precision/recall and alert-volume curves.

Terminology: a "false positive" here means an alert on a transaction whose
synthetic label is 0. It is a false positive *relative to the simulated label*,
not proof that the real-world transaction would be harmless.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def confusion_counts(y_true, flagged) -> dict:
    y = np.asarray(y_true).astype(bool)
    f = np.asarray(flagged).astype(bool)
    return {
        "tp": int(np.sum(y & f)),
        "fp": int(np.sum(~y & f)),
        "fn": int(np.sum(y & ~f)),
        "tn": int(np.sum(~y & ~f)),
    }


def summarize_alerts(y_true, flagged) -> dict:
    """Alert volume and quality for one set of binary alerts."""
    c = confusion_counts(y_true, flagged)
    n = sum(c.values())
    alerts = c["tp"] + c["fp"]
    positives = c["tp"] + c["fn"]
    precision = c["tp"] / alerts if alerts else float("nan")
    recall = c["tp"] / positives if positives else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if alerts and positives and (precision + recall) > 0
        else 0.0
    )
    prevalence = positives / n if n else float("nan")
    return {
        "n_transactions": n,
        "n_alerts": alerts,
        "alert_rate": alerts / n if n else float("nan"),
        **c,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # How many negative-label alerts an analyst reviews per labelled hit.
        "fp_per_tp": c["fp"] / c["tp"] if c["tp"] else float("inf"),
        # Precision relative to flagging transactions at random.
        "lift": precision / prevalence if alerts and prevalence else float("nan"),
    }


def alert_curve(y_true, scores) -> pd.DataFrame:
    """Alert volume, recall and precision for every possible threshold.

    Rows are ordered from most restrictive (highest threshold) to most
    sensitive. Tied scores are handled together, because a threshold cannot
    split them. Scores of -inf (e.g. "no history") never alert.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s)
    frame = pd.DataFrame({"score": s[keep], "y": y[keep]})
    grouped = frame.groupby("score")["y"].agg(["sum", "count"]).sort_index(ascending=False)
    tp = grouped["sum"].cumsum().to_numpy()
    alerts = grouped["count"].cumsum().to_numpy()
    total_pos = y.sum()
    return pd.DataFrame(
        {
            "threshold": grouped.index.to_numpy(),
            "n_alerts": alerts,
            "alert_rate": alerts / len(y),
            "tp": tp,
            "fp": alerts - tp,
            "recall": tp / total_pos if total_pos else np.nan,
            "precision": tp / alerts,
        }
    )


def operating_points(name, val_scores, test_scores, y_test, budgets: dict, threshold_fn) -> pd.DataFrame:
    """Choose one threshold per alert budget on the validation scores, then
    apply that fixed threshold to the test scores and summarise the alerts."""
    rows = []
    for setting, budget in budgets.items():
        t = threshold_fn(val_scores, budget)
        flagged = np.asarray(test_scores, dtype=float) >= t
        val_rate = float(np.mean(np.asarray(val_scores, dtype=float) >= t))
        rows.append({"approach": name, "setting": setting, "budget": budget, "threshold": t,
                     "validation_alert_rate": val_rate, **summarize_alerts(y_test, flagged)})
    return pd.DataFrame(rows)


def pr_auc(y_true, scores) -> float:
    """Average precision (area under the precision-recall curve).

    -inf scores are replaced by a value below every finite score so that
    sklearn can rank them (they end up last).
    """
    s = np.asarray(scores, dtype=float).copy()
    finite = s[np.isfinite(s)]
    floor = (finite.min() - 1.0) if finite.size else 0.0
    s[~np.isfinite(s)] = floor
    return float(average_precision_score(np.asarray(y_true).astype(int), s))
