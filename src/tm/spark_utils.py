"""Create a local SparkSession with settings that work on a laptop."""
import os
import sys

from pyspark.sql import SparkSession


def get_spark(app_name: str = "transaction-monitoring", driver_memory: str = "6g") -> SparkSession:
    # Make sure Spark's Python workers use the same interpreter as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        # The feature table (~5M rows) is collected to pandas once; allow >1 GB.
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
