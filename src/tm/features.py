"""Historical account-behaviour features computed with Spark window functions.

Every feature describes what was known *before* the transaction happened.
Timestamps in the data have minute resolution, so transactions in the same
minute cannot be ordered. The windows therefore end one second before the
current transaction (rangeBetween(-window, -1)): other transactions from the
same minute, and of course all later ones, are never used.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _past_window(partition_col: str, seconds: int):
    return Window.partitionBy(partition_col).orderBy("ts_seconds").rangeBetween(-seconds, -1)


def add_sender_history(df: DataFrame, window_24h: int, window_7d: int, min_history: int) -> DataFrame:
    """Behaviour of the *paying* account before this transaction."""
    w24 = _past_window("from_account", window_24h)
    w7 = _past_window("from_account", window_7d)
    df = (
        df.withColumn("sender_n_out_24h", F.count("*").over(w24))
        .withColumn("sender_n_recipients_24h", F.size(F.collect_set("to_account").over(w24)))
        .withColumn("sender_n_out_7d", F.count("*").over(w7))
        .withColumn("sender_mean_out_usd_7d", F.avg("amount_usd").over(w7))
    )
    # Ratio of this amount to the account's typical amount; only defined when
    # the account has enough history to have a "typical" amount.
    return df.withColumn(
        "amount_vs_history",
        F.when(
            (F.col("sender_n_out_7d") >= min_history) & (F.col("sender_mean_out_usd_7d") > 0),
            F.col("amount_usd") / F.col("sender_mean_out_usd_7d"),
        ),
    )


def add_receiver_history(df: DataFrame, window_24h: int) -> DataFrame:
    """Behaviour of the *receiving* account before this transaction."""
    w24 = _past_window("to_account", window_24h)
    return df.withColumn("receiver_n_in_24h", F.count("*").over(w24)).withColumn(
        "receiver_n_senders_24h", F.size(F.collect_set("from_account").over(w24))
    )
