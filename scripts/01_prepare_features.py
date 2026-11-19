"""Step 1 (PySpark): load, clean, convert to USD, build history features.

Output: data/processed/features.parquet (one row per transaction, 1-10 Sept)
        results/tables/usd_rates.csv, results/tables/split_summary.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from tm import config  # noqa: E402
from tm.features import add_receiver_history, add_sender_history  # noqa: E402
from tm.load import (  # noqa: E402
    add_amount_usd,
    assign_split,
    clean_transactions,
    estimate_usd_rates,
    read_transactions,
    restrict_period,
)
from tm.spark_utils import get_spark  # noqa: E402

KEEP_COLUMNS = [
    "timestamp", "timestamp_str", "from_bank", "from_acct", "to_bank", "to_acct",
    "from_account", "to_account", "amount_paid_str", "payment_currency", "receiving_currency",
    "payment_format", "amount_usd", "is_self_transfer", "is_cross_bank", "is_cross_currency",
    "sender_n_out_24h", "sender_n_recipients_24h", "sender_n_out_7d", "sender_mean_out_usd_7d",
    "amount_vs_history", "receiver_n_in_24h", "receiver_n_senders_24h",
    "split", "label",
]
SORT_KEYS = [
    "timestamp", "from_account", "to_account", "amount_paid_str", "payment_currency",
    "receiving_currency", "payment_format", "label",
]


def main() -> None:
    if not config.RAW_TRANSACTIONS.exists():
        sys.exit(f"Missing {config.RAW_TRANSACTIONS}. Run scripts/00_download_data.py first.")
    start = time.time()
    spark = get_spark("01_prepare_features")
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = read_transactions(spark, config.RAW_TRANSACTIONS)
    clean = clean_transactions(raw)
    n_raw = clean.count()

    df = restrict_period(clean, config.ANALYSIS_START, config.ANALYSIS_END)
    df = assign_split(df, config.SPLITS).cache()
    print(f"rows in file: {n_raw:,}; rows 1-10 Sept: {df.count():,}")

    # Exchange rates are estimated on the training period only.
    rates = estimate_usd_rates(df.filter(F.col("split") == "train"))
    pd.Series(rates, name="usd_per_unit").rename_axis("currency").to_csv(config.TABLES_DIR / "usd_rates.csv")
    print("estimated USD rates:", {k: round(v, 4) for k, v in sorted(rates.items())})

    df = add_amount_usd(df, rates)
    df = add_sender_history(df, config.WINDOW_24H, config.WINDOW_7D, config.MIN_HISTORY_TXNS)
    df = add_receiver_history(df, config.WINDOW_24H)

    pdf = df.select(*KEEP_COLUMNS).toPandas()
    # Deterministic row order and a stable row identifier. Later steps join
    # model scores back to transactions on `row_id`, never on position alone.
    pdf = pdf.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)
    pdf.insert(0, "row_id", np.arange(len(pdf), dtype=np.int64))
    # Microsecond timestamps so that Spark can read the file again in step 2.
    pdf.to_parquet(config.FEATURES_FILE, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)

    summary = (
        pdf.groupby("split", sort=False)
        .agg(first_day=("timestamp", "min"), last_day=("timestamp", "max"),
             transactions=("label", "size"), positive_labels=("label", "sum"))
        .assign(positive_rate=lambda d: d["positive_labels"] / d["transactions"])
    )
    summary.to_csv(config.TABLES_DIR / "split_summary.csv")
    print(summary)
    print(f"saved {len(pdf):,} rows to {config.FEATURES_FILE} in {time.time() - start:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
