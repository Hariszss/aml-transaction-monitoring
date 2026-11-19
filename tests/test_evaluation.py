import math

import numpy as np
import pytest

from tm.evaluation import alert_curve, confusion_counts, pr_auc, summarize_alerts


def test_confusion_counts():
    y = [1, 1, 0, 0, 0]
    f = [1, 0, 1, 0, 0]
    assert confusion_counts(y, f) == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}


def test_summarize_alerts_metrics():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    f = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    s = summarize_alerts(y, f)
    assert s["n_alerts"] == 3 and s["alert_rate"] == pytest.approx(0.3)
    assert s["precision"] == pytest.approx(1 / 3)
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(2 * (1 / 3) * 0.5 / (1 / 3 + 0.5))
    assert s["fp_per_tp"] == pytest.approx(2.0)
    assert s["lift"] == pytest.approx((1 / 3) / 0.2)


def test_summarize_alerts_without_alerts():
    s = summarize_alerts([1, 0, 0], [0, 0, 0])
    assert s["n_alerts"] == 0 and s["recall"] == 0
    assert math.isnan(s["precision"]) and s["f1"] == 0.0
    assert s["fp_per_tp"] == float("inf")


def test_alert_curve_handles_ties_and_missing():
    y = [1, 0, 1, 0, 0]
    scores = [5.0, 5.0, 3.0, 1.0, -np.inf]
    c = alert_curve(y, scores)
    assert list(c["threshold"]) == [5.0, 3.0, 1.0]
    assert list(c["n_alerts"]) == [2, 3, 4]  # the two tied 5.0 scores enter together
    assert list(c["tp"]) == [1, 2, 2]
    assert c["recall"].iloc[-1] == pytest.approx(1.0)
    assert c["alert_rate"].iloc[-1] == pytest.approx(0.8)  # the -inf row never alerts


def test_pr_auc_perfect_and_with_missing_scores():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # -inf (no score) must not crash and is ranked last.
    assert pr_auc([1, 0, 0], [2.0, 1.0, -np.inf]) == pytest.approx(1.0)
