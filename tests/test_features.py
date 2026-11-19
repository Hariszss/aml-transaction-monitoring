"""The key property: history features never use the current transaction,
transactions in the same minute, or anything later."""
import pandas as pd
import pytest

from tm.features import add_receiver_history, add_sender_history

H = 3600
COLS = ["txn", "ts_seconds", "from_account", "to_account", "amount_usd"]


def _features(spark, rows, min_history=2):
    df = spark.createDataFrame(pd.DataFrame(rows, columns=COLS))
    df = add_sender_history(df, window_24h=24 * H, window_7d=7 * 24 * H, min_history=min_history)
    df = add_receiver_history(df, window_24h=24 * H)
    return {r["txn"]: r for r in df.collect()}


def test_first_transaction_has_no_history(spark):
    out = _features(spark, [(1, 0, "S", "R1", 100.0)])
    assert out[1]["sender_n_out_24h"] == 0
    assert out[1]["sender_n_recipients_24h"] == 0
    assert out[1]["amount_vs_history"] is None


def test_only_past_transactions_are_counted(spark):
    rows = [
        (1, 0, "S", "R1", 100.0),
        (2, 1 * H, "S", "R2", 100.0),
        (3, 2 * H, "S", "R2", 400.0),   # the transaction we check
        (4, 3 * H, "S", "R3", 9999.0),  # future: must not affect txn 3
    ]
    out = _features(spark, rows)
    assert out[3]["sender_n_out_24h"] == 2
    assert out[3]["sender_n_recipients_24h"] == 2  # R1, R2
    assert out[3]["sender_mean_out_usd_7d"] == pytest.approx(100.0)
    assert out[3]["amount_vs_history"] == pytest.approx(4.0)


def test_same_minute_transactions_are_excluded(spark):
    # Two payments in the same minute cannot be ordered, so neither sees the other.
    rows = [(1, 600, "S", "R1", 10.0), (2, 600, "S", "R2", 10.0)]
    out = _features(spark, rows)
    assert out[1]["sender_n_out_24h"] == 0
    assert out[2]["sender_n_out_24h"] == 0


def test_24h_window_boundary(spark):
    rows = [
        (1, 0, "S", "R1", 10.0),
        (2, 24 * H, "S", "R2", 10.0),      # exactly 24h later: txn 1 still inside
        (3, 24 * H + 61, "S", "R3", 10.0),  # just over 24h after txn 1
    ]
    out = _features(spark, rows)
    assert out[2]["sender_n_out_24h"] == 1
    assert out[3]["sender_n_out_24h"] == 1  # only txn 2


def test_history_ratio_needs_minimum_history(spark):
    rows = [(1, 0, "S", "R1", 10.0), (2, H, "S", "R1", 50.0)]
    out = _features(spark, rows, min_history=2)
    assert out[2]["sender_n_out_7d"] == 1
    assert out[2]["amount_vs_history"] is None


def test_receiver_fan_in(spark):
    rows = [
        (1, 0, "A", "R", 10.0),
        (2, H, "B", "R", 10.0),
        (3, 2 * H, "C", "R", 10.0),
        (4, 3 * H, "A", "R", 10.0),
    ]
    out = _features(spark, rows)
    assert out[4]["receiver_n_in_24h"] == 3
    assert out[4]["receiver_n_senders_24h"] == 3  # A, B, C
    # Sender A paid R once before: one distinct recipient, not three.
    assert out[4]["sender_n_recipients_24h"] == 1
