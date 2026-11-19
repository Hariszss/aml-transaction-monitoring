"""Loading and cleaning the IBM AML transaction file with PySpark."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

# The CSV header repeats "Account" twice, so we read every column as a string
# with our own names and cast explicitly afterwards.
RAW_COLUMNS = [
    "timestamp_str",
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
RAW_SCHEMA = StructType([StructField(c, StringType(), True) for c in RAW_COLUMNS])


def read_transactions(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(str(path), header=True, schema=RAW_SCHEMA)


def clean_transactions(raw: DataFrame) -> DataFrame:
    """Parse types and build account identifiers.

    Account numbers are only unique together with the bank code, so an
    account is identified as "<bank>_<account>".
    """
    df = (
        raw.withColumn("amount_paid_str", F.col("amount_paid"))  # kept for joining with patterns
        .withColumn("timestamp", F.to_timestamp("timestamp_str", "yyyy/MM/dd HH:mm"))
        .withColumn("ts_seconds", F.unix_timestamp("timestamp"))
        .withColumn("amount_paid", F.col("amount_paid").cast("double"))
        .withColumn("amount_received", F.col("amount_received").cast("double"))
        .withColumn("label", F.col("is_laundering").cast("int"))
        .withColumn("from_account", F.concat_ws("_", "from_bank", "from_acct"))
        .withColumn("to_account", F.concat_ws("_", "to_bank", "to_acct"))
        .withColumn("is_self_transfer", (F.col("from_account") == F.col("to_account")).cast("int"))
        .withColumn("is_cross_bank", (F.col("from_bank") != F.col("to_bank")).cast("int"))
        .withColumn(
            "is_cross_currency",
            (F.col("payment_currency") != F.col("receiving_currency")).cast("int"),
        )
    )
    # Drop rows that could not be parsed (none are expected, but be explicit).
    return df.dropna(subset=["timestamp", "amount_paid", "label", "from_account", "to_account"])


def restrict_period(df: DataFrame, start: str, end_inclusive: str) -> DataFrame:
    day = F.to_date("timestamp")
    return df.filter((day >= F.lit(start)) & (day <= F.lit(end_inclusive)))


def estimate_usd_rates(df: DataFrame) -> dict:
    """Estimate one exchange rate per currency (USD per unit) from the data.

    The file contains no rate table, but cross-currency payments record both
    the amount paid and the amount received. For every currency X we take the
    median implied rate over payments X -> USD and USD -> X.
    """
    to_usd = df.filter(
        (F.col("receiving_currency") == "US Dollar") & (F.col("payment_currency") != "US Dollar")
    ).select(
        F.col("payment_currency").alias("currency"),
        (F.col("amount_received") / F.col("amount_paid")).alias("usd_per_unit"),
    )
    from_usd = df.filter(
        (F.col("payment_currency") == "US Dollar") & (F.col("receiving_currency") != "US Dollar")
    ).select(
        F.col("receiving_currency").alias("currency"),
        (F.col("amount_paid") / F.col("amount_received")).alias("usd_per_unit"),
    )
    pairs = to_usd.unionByName(from_usd).filter(
        F.col("usd_per_unit").isNotNull() & (F.col("usd_per_unit") > 0)
    )
    rows = (
        pairs.groupBy("currency")
        .agg(F.percentile_approx("usd_per_unit", 0.5, 10000).alias("rate"), F.count("*").alias("n"))
        .collect()
    )
    rates = {r["currency"]: float(r["rate"]) for r in rows}
    rates["US Dollar"] = 1.0
    return rates


def add_amount_usd(df: DataFrame, rates: dict) -> DataFrame:
    """Convert the paid amount to USD using a fixed rate per currency."""
    mapping = F.create_map(*[x for c, r in rates.items() for x in (F.lit(c), F.lit(r))])
    return df.withColumn("amount_usd", F.col("amount_paid") * mapping[F.col("payment_currency")])


def assign_split(df: DataFrame, splits: dict) -> DataFrame:
    """Label every row with the chronological period it belongs to."""
    day = F.to_date("timestamp")
    expr = None
    for name, (start, end) in splits.items():
        cond = (day >= F.lit(start)) & (day <= F.lit(end))
        expr = F.when(cond, F.lit(name)) if expr is None else expr.when(cond, F.lit(name))
    return df.withColumn("split", expr)
