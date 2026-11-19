import numpy as np
import pandas as pd
import pytest

from tm.model import attach_lr_scores, build_pipeline, make_model_frame
from tm.patterns import parse_patterns
from tm.rules import apply_threshold, rule_scores, threshold_for_alert_budget


def test_threshold_respects_budget():
    scores = np.arange(1, 101, dtype=float)  # 100 distinct scores
    t = threshold_for_alert_budget(scores, budget=0.05)
    assert t == 96.0
    assert apply_threshold(scores, t).sum() == 5


def test_threshold_with_ties_stays_within_budget():
    scores = np.array([1.0] * 90 + [5.0] * 8 + [9.0] * 2)
    # 5% budget = 5 alerts; including the tied 5.0 scores would give 10 alerts.
    t = threshold_for_alert_budget(scores, budget=0.05)
    assert t == 9.0
    assert apply_threshold(scores, t).sum() == 2


def test_threshold_impossible_budget_returns_inf():
    scores = np.array([3.0] * 10)
    assert threshold_for_alert_budget(scores, budget=0.05) == float("inf")


def test_missing_rule_scores_never_alert():
    df = pd.DataFrame({"amount_vs_history": [np.nan, 10.0]})
    s = rule_scores(df, "amount_vs_history")
    assert list(apply_threshold(s, 0.0)) == [False, True]


def test_parse_patterns():
    text = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops
2022/09/02 01:00,001,A,002,B,10.00,US Dollar,10.00,US Dollar,ACH,1
2022/09/02 02:00,002,B,001,A,9.00,US Dollar,9.00,US Dollar,ACH,1
END LAUNDERING ATTEMPT - CYCLE
""".splitlines()
    p = parse_patterns(text)
    assert list(p["typology"]) == ["FAN-OUT", "CYCLE", "CYCLE"]
    assert list(p["attempt_id"]) == [1, 2, 2]
    assert p.loc[0, "amount_paid_str"] == "2848.96" and p.loc[0, "from_bank"] == "021174"


def test_model_preprocessing_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    n = 200
    raw = pd.DataFrame(
        {
            "amount_usd": rng.lognormal(5, 1, n),
            "amount_vs_history": np.where(rng.random(n) < 0.3, np.nan, rng.lognormal(0, 1, n)),
            "sender_n_out_24h": rng.integers(0, 5, n),
            "sender_n_recipients_24h": rng.integers(0, 5, n),
            "receiver_n_senders_24h": rng.integers(0, 5, n),
            "sender_n_out_7d": rng.integers(0, 20, n),
            "is_cross_bank": rng.integers(0, 2, n),
            "is_cross_currency": rng.integers(0, 2, n),
            "is_self_transfer": rng.integers(0, 2, n),
            "payment_format": rng.choice(["ACH", "Cheque"], n),
        }
    )
    y = rng.integers(0, 2, n)
    X = make_model_frame(raw)
    assert X.isna().sum().sum() == 0
    pipe = build_pipeline().fit(X.iloc[:100], y[:100])
    scaler = pipe.named_steps["pre"].named_transformers_["num"]
    # Scaler means equal the training-half means, not the full-data means.
    assert scaler.mean_[0] == pytest.approx(X["log_amount_usd"].iloc[:100].mean())
    # Unseen category at prediction time is ignored rather than crashing.
    X_new = X.iloc[100:].copy()
    X_new.loc[X_new.index[0], "payment_format"] = "Wire"
    assert pipe.predict_proba(X_new).shape == (100, 2)


def test_scores_are_joined_on_row_id_not_position():
    tx = pd.DataFrame({"row_id": [10, 11, 12], "label": [0, 1, 0]})
    scores = pd.DataFrame({"row_id": [12, 10, 11], "label": [0, 0, 1], "lr_score": [0.2, 0.0, 0.1]})
    out = attach_lr_scores(tx, scores)  # scores deliberately in a different order
    assert list(out["lr_score"]) == [0.0, 0.1, 0.2]


def test_score_join_rejects_missing_or_mismatched_rows():
    tx = pd.DataFrame({"row_id": [1, 2], "label": [0, 1]})
    with pytest.raises(ValueError):
        attach_lr_scores(tx, pd.DataFrame({"row_id": [1], "label": [0], "lr_score": [0.5]}))
    with pytest.raises(ValueError):
        attach_lr_scores(tx, pd.DataFrame({"row_id": [1, 2], "label": [0, 0], "lr_score": [0.5, 0.6]}))
    with pytest.raises(pd.errors.MergeError):
        attach_lr_scores(tx, pd.DataFrame({"row_id": [1, 1, 2], "label": [0, 0, 1], "lr_score": [0.1, 0.2, 0.3]}))
