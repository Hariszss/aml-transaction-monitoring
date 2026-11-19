import pandas as pd
import pytest

from tm.load import (
    RAW_COLUMNS,
    add_amount_usd,
    assign_split,
    clean_transactions,
    estimate_usd_rates,
    restrict_period,
)

ROWS = [
    # ts, from_bank, from_acct, to_bank, to_acct, amt_rec, rec_cur, amt_paid, pay_cur, fmt, label
    ("2022/09/01 00:20", "010", "A1", "010", "A1", "100.0", "US Dollar", "100.0", "US Dollar", "Reinvestment", "0"),
    ("2022/09/01 10:00", "010", "A1", "020", "B1", "50.0", "Euro", "50.0", "Euro", "ACH", "1"),
    ("2022/09/02 10:00", "020", "B1", "010", "A1", "220.0", "US Dollar", "200.0", "Euro", "Wire", "0"),
    ("2022/09/11 10:00", "030", "C1", "010", "A1", "10.0", "US Dollar", "10.0", "US Dollar", "ACH", "1"),
]


@pytest.fixture
def clean_df(spark):
    raw = spark.createDataFrame(pd.DataFrame(ROWS, columns=RAW_COLUMNS))
    return clean_transactions(raw)


def test_clean_parses_types_and_accounts(clean_df):
    rows = clean_df.orderBy("timestamp").collect()
    assert rows[0]["label"] == 0 and rows[1]["label"] == 1
    assert rows[0]["amount_paid"] == pytest.approx(100.0)
    # Account identifiers include the bank code.
    assert rows[1]["from_account"] == "010_A1" and rows[1]["to_account"] == "020_B1"
    assert rows[0]["is_self_transfer"] == 1 and rows[1]["is_self_transfer"] == 0
    assert rows[1]["is_cross_bank"] == 1 and rows[0]["is_cross_bank"] == 0
    assert rows[2]["is_cross_currency"] == 1 and rows[1]["is_cross_currency"] == 0
    # Minute-resolution timestamps become epoch seconds.
    assert rows[1]["ts_seconds"] - rows[0]["ts_seconds"] == (9 * 60 + 40) * 60


def test_restrict_period_is_inclusive(clean_df):
    kept = restrict_period(clean_df, "2022-09-01", "2022-09-10")
    assert kept.count() == 3


def test_assign_split(clean_df):
    splits = {"train": ("2022-09-01", "2022-09-01"), "test": ("2022-09-02", "2022-09-10")}
    got = {r["timestamp_str"]: r["split"] for r in assign_split(clean_df, splits).collect()}
    assert got["2022/09/01 10:00"] == "train"
    assert got["2022/09/02 10:00"] == "test"
    assert got["2022/09/11 10:00"] is None


def test_usd_rate_estimation_and_conversion(clean_df):
    rates = estimate_usd_rates(clean_df)
    # One Euro->USD payment: 200 EUR paid, 220 USD received -> 1.1 USD per EUR.
    assert rates["Euro"] == pytest.approx(1.1)
    assert rates["US Dollar"] == 1.0
    converted = {r["timestamp_str"]: r["amount_usd"] for r in add_amount_usd(clean_df, rates).collect()}
    assert converted["2022/09/01 10:00"] == pytest.approx(55.0)
    assert converted["2022/09/01 00:20"] == pytest.approx(100.0)
