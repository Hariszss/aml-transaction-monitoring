"""Three transparent monitoring rules and threshold selection.

Each rule turns one observable quantity into a score; a transaction raises an
alert when score >= threshold. Thresholds are chosen on the validation period
only, so that the test period is used once, for the final comparison.

These rules are illustrative choices made for this project. They are not
any bank's monitoring rules.
"""
import numpy as np
import pandas as pd

# rule name -> (score column, short description)
RULES = {
    "large_amount": (
        "amount_usd",
        "Payment amount (USD) is at or above a threshold",
    ),
    "amount_vs_history": (
        "amount_vs_history",
        "Payment is at least k times the sender's average payment in the previous 7 days "
        "(needs >= 3 earlier payments)",
    ),
    "fan_in": (
        "receiver_n_senders_24h",
        "Receiving account was paid by at least n different accounts in the previous 24 hours",
    ),
}


def rule_scores(df: pd.DataFrame, rule: str) -> np.ndarray:
    """Return the rule's score; missing values (e.g. no history) never alert."""
    col = RULES[rule][0]
    return df[col].astype(float).fillna(-np.inf).to_numpy()


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores) >= threshold


def threshold_for_alert_budget(scores: np.ndarray, budget: float) -> float:
    """Lowest threshold whose alert rate does not exceed `budget`.

    `budget` is a share of all transactions (e.g. 0.01 = flag at most 1%).
    Because many scores are tied (counts, round amounts), the achieved alert
    rate can be lower than the budget. Returns +inf if even the highest score
    value would exceed the budget.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    finite = scores[np.isfinite(scores)]
    values, counts = np.unique(finite, return_counts=True)
    # Walk from the highest value down; alerts(t) = number of scores >= t.
    alerts_at = np.cumsum(counts[::-1])  # alerts if threshold = values[::-1][i]
    allowed = alerts_at <= budget * n
    if not allowed.any():
        return float("inf")
    last_ok = np.nonzero(allowed)[0].max()
    return float(values[::-1][last_ok])
