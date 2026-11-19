"""Step 2 (PySpark + Spark SQL): exploratory analysis.

Only the training and validation periods (1-7 Sept) are explored, so that
what we learn here (and use to design the rules) is not influenced by the
held-out test period (8-10 Sept).

Outputs: results/tables/eda_*.csv, figures/fig1_*.png, figures/fig2_*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from tm import config, plotting  # noqa: E402
from tm.spark_utils import get_spark  # noqa: E402

plt = plotting.plt


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    print(f"\n== {name} ==\n{df.to_string(index=False)}")
    return df


def main() -> None:
    spark = get_spark("02_explore")
    tx = (
        spark.read.parquet(str(config.FEATURES_FILE))
        .filter(F.col("split").isin("train", "validation"))
        .cache()
    )
    tx.createOrReplaceTempView("tx")
    print(f"rows explored (1-7 Sept): {tx.count():,}")

    # 1. Class imbalance.
    save(spark.sql("""
        SELECT label, COUNT(*) AS transactions,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct
        FROM tx GROUP BY label ORDER BY label
    """).toPandas(), "eda_class_balance")

    # 2. Spark SQL: payment format - how transactions and positive labels are distributed.
    fmt = save(spark.sql("""
        SELECT payment_format,
               COUNT(*)                                   AS transactions,
               SUM(label)                                 AS positive_labels,
               ROUND(100 * AVG(label), 3)                 AS positive_rate_pct,
               ROUND(100 * COUNT(*)  / SUM(COUNT(*))  OVER (), 2) AS pct_of_transactions,
               ROUND(100 * SUM(label) / SUM(SUM(label)) OVER (), 2) AS pct_of_positive_labels
        FROM tx
        GROUP BY payment_format
        ORDER BY pct_of_positive_labels DESC
    """).toPandas(), "eda_payment_format")

    # 3. Amounts: distribution by label (USD).
    amt = (
        tx.groupBy("label")
        .agg(
            F.count("*").alias("n"),
            *[F.percentile_approx("amount_usd", q, 10000).alias(f"p{int(q * 100)}")
              for q in (0.10, 0.25, 0.5, 0.75, 0.90, 0.99)],
        )
        .orderBy("label")
        .toPandas()
    )
    save(amt.round(2), "eda_amount_quantiles_by_label")

    # Positive rate per order of magnitude of the amount.
    amount_band = (
        tx.withColumn("band", F.floor(F.log10(F.greatest(F.col("amount_usd"), F.lit(1.0)))))
        .groupBy("band")
        .agg(F.count("*").alias("transactions"), F.sum("label").alias("positive_labels"))
        .withColumn("positive_rate_pct", F.round(100 * F.col("positive_labels") / F.col("transactions"), 3))
        .orderBy("band")
        .toPandas()
    )
    save(amount_band, "eda_amount_band")

    # 4. Account activity and counterparties: history features by label.
    hist = (
        tx.groupBy("label")
        .agg(
            F.round(F.avg("sender_n_out_24h"), 2).alias("mean_sender_n_out_24h"),
            F.round(F.avg("sender_n_recipients_24h"), 2).alias("mean_sender_n_recipients_24h"),
            F.round(F.avg("receiver_n_senders_24h"), 2).alias("mean_receiver_n_senders_24h"),
            F.round(100 * F.avg(F.col("amount_vs_history").isNotNull().cast("int")), 1).alias("pct_with_7d_history"),
            F.percentile_approx("amount_vs_history", 0.5, 10000).alias("median_amount_vs_history"),
            F.round(100 * F.avg("is_cross_bank"), 1).alias("pct_cross_bank"),
            F.round(100 * F.avg("is_cross_currency"), 2).alias("pct_cross_currency"),
            F.round(100 * F.avg("is_self_transfer"), 2).alias("pct_self_transfer"),
        )
        .orderBy("label")
        .toPandas()
    )
    save(hist, "eda_history_features_by_label")

    # Counterparty counts in the previous 24 hours, bucketed.
    def bucketed(col: str, name: str) -> pd.DataFrame:
        return save(spark.sql(f"""
            SELECT CASE WHEN {col} >= 10 THEN '10+' ELSE CAST({col} AS STRING) END AS bucket,
                   MIN({col}) AS sort_key,
                   COUNT(*) AS transactions, SUM(label) AS positive_labels,
                   ROUND(100 * AVG(label), 3) AS positive_rate_pct
            FROM tx GROUP BY 1 ORDER BY sort_key
        """).toPandas(), name)

    bucketed("sender_n_recipients_24h", "eda_sender_recipients_24h")
    fan_in = bucketed("receiver_n_senders_24h", "eda_receiver_senders_24h")

    # Quiet vs busy senders: how much of each label comes from accounts that
    # made no payment in the previous 24h, and from very busy accounts?
    save(spark.sql("""
        SELECT CASE WHEN sender_n_out_24h = 0 THEN 'a) no payment in previous 24h'
                    WHEN sender_n_out_24h <= 10 THEN 'b) 1-10 payments'
                    WHEN sender_n_out_24h <= 100 THEN 'c) 11-100 payments'
                    ELSE 'd) more than 100 payments' END AS sender_activity_24h,
               COUNT(*) AS transactions, SUM(label) AS positive_labels,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_transactions,
               ROUND(100 * SUM(label) / SUM(SUM(label)) OVER (), 2) AS pct_of_positive_labels,
               ROUND(100 * AVG(label), 3) AS positive_rate_pct
        FROM tx GROUP BY 1 ORDER BY 1
    """).toPandas(), "eda_sender_activity")

    # 5. Spark SQL: incoming vs outgoing activity of accounts that send labelled
    # transactions. Do such accounts also send many *unlabelled* payments?
    save(spark.sql("""
        WITH sender AS (
            SELECT from_account AS account, COUNT(*) AS n_out, SUM(label) AS n_out_pos
            FROM tx WHERE is_self_transfer = 0 GROUP BY from_account
        ),
        receiver AS (
            SELECT to_account AS account, COUNT(*) AS n_in, SUM(label) AS n_in_pos
            FROM tx WHERE is_self_transfer = 0 GROUP BY to_account
        ),
        acct AS (
            SELECT COALESCE(s.account, r.account) AS account,
                   COALESCE(n_out, 0) AS n_out, COALESCE(n_out_pos, 0) AS n_out_pos,
                   COALESCE(n_in, 0) AS n_in, COALESCE(n_in_pos, 0) AS n_in_pos
            FROM sender s FULL OUTER JOIN receiver r ON s.account = r.account
        )
        SELECT CASE WHEN n_out_pos + n_in_pos > 0 THEN 'involved in >=1 labelled txn'
                    ELSE 'no labelled txn' END AS account_group,
               COUNT(*) AS accounts,
               ROUND(AVG(n_out), 1) AS mean_outgoing,
               ROUND(AVG(n_in), 1) AS mean_incoming,
               ROUND(100 * SUM(n_out_pos) / NULLIF(SUM(n_out), 0), 2) AS pct_outgoing_labelled,
               ROUND(100 * SUM(n_in_pos) / NULLIF(SUM(n_in), 0), 2) AS pct_incoming_labelled
        FROM acct GROUP BY 1 ORDER BY 1
    """).toPandas(), "eda_accounts_in_out")

    # 6. Daily volume (shows the weekly rhythm of the simulation).
    save(spark.sql("""
        SELECT CAST(TO_DATE(timestamp) AS STRING) AS day, COUNT(*) AS transactions,
               SUM(label) AS positive_labels, ROUND(100 * AVG(label), 3) AS positive_rate_pct
        FROM tx GROUP BY 1 ORDER BY 1
    """).toPandas(), "eda_daily")

    spark.stop()
    make_figures(fmt, amount_band, fan_in)


def make_figures(fmt: pd.DataFrame, amount_band: pd.DataFrame, fan_in: pd.DataFrame) -> None:
    plotting.setup()
    config.FIGURES_DIR.mkdir(exist_ok=True)

    # Figure 1: share of transactions vs share of positive labels per payment format.
    f = fmt.sort_values("pct_of_transactions")
    y = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    h = 0.38
    ax.barh(y + h / 2, f["pct_of_transactions"], height=h - 0.04, color=plotting.NEUTRAL, label="Share of all transactions")
    ax.barh(y - h / 2, f["pct_of_positive_labels"], height=h - 0.04, color="#2a78d6", label="Share of positive labels")
    for yi, (a, b) in enumerate(zip(f["pct_of_transactions"], f["pct_of_positive_labels"])):
        ax.text(a + 0.8, yi + h / 2, f"{a:.1f}%", va="center", fontsize=8, color=plotting.TEXT_SECONDARY)
        ax.text(b + 0.8, yi - h / 2, f"{b:.1f}%", va="center", fontsize=8, color=plotting.TEXT)
    ax.set_yticks(y, f["payment_format"])
    ax.set_xlabel("% (1-7 Sept, train + validation)")
    ax.set_title("Share of transactions and of positive labels by payment format")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig1_payment_format.png")
    plt.close(fig)

    # Figure 2: positive-label rate by amount band and by receiver fan-in.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    base_rate = 100 * fan_in["positive_labels"].sum() / fan_in["transactions"].sum()
    a = amount_band[amount_band["transactions"] >= 200]
    labels = [f"1e{int(b)}" for b in a["band"]]
    axes[0].bar(labels, a["positive_rate_pct"], color="#2a78d6", width=0.7)
    axes[0].set_xlabel("Amount (USD), order of magnitude (1e3 = 1,000-9,999)")
    axes[0].set_ylabel("% of transactions with positive label")
    axes[0].set_title("By payment amount")
    axes[1].bar(fan_in["bucket"], fan_in["positive_rate_pct"], color="#1baf7a", width=0.7)
    axes[1].set_xlabel("Distinct accounts that paid the receiver in previous 24h")
    axes[1].set_title("By receiver fan-in")
    for ax in axes:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.axhline(base_rate, color=plotting.TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.grid(axis="x", visible=False)
    axes[0].text(0, base_rate * 1.15, f"overall rate {base_rate:.2f}%", fontsize=8, color=plotting.TEXT_SECONDARY)
    fig.suptitle("Positive-label rate by payment amount and by receiver fan-in (1-7 Sept)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig2_rate_by_amount_and_fan_in.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
